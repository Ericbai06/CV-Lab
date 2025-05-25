#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于 VideoMAE 的视频分类模型微调脚本
结合了数据集处理和模型训练功能
"""

import os
import torch
import numpy as np
import cv2
import av
import logging
import sys
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import evaluate
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    Normalize
)
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置环境变量来配置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 检查CUDA是否可用
device = torch.device("cuda")
logger.info(f"使用设备: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
    logger.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 配置参数
class Config:
    BATCH_SIZE = 8
    MAX_FRAMES = 16  # VideoMAE 默认帧数
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2
    MIXED_PRECISION = True
    MODEL_CKPT = "./models--MCG-NJU--videomae-base/snapshots/base-model"
    IMAGE_SIZE = 224  # 视频帧大小

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor, max_frames=Config.MAX_FRAMES, preload=True, num_workers=16):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.max_frames = max_frames
        self.preload = preload
        self.cached_videos = {}
        self.cache_lock = threading.Lock()
        self.num_workers = num_workers
        logger.info(f"初始化数据集，共 {len(video_paths)} 个视频")

        if preload:
            logger.info(f"开始使用 {num_workers} 个线程预加载视频到内存...")
            self._preload_videos()
            logger.info("视频预加载完成")

    def _preload_videos(self):
        """使用多线程预加载所有视频"""
        def load_worker(idx):
            try:
                frames_tensor, label = self._load_video(idx)
                with self.cache_lock:
                    self.cached_videos[idx] = (frames_tensor, label)
                return idx
            except Exception as e:
                logger.error(f"线程加载视频 {idx} 失败: {str(e)}")
                return None
                
        # 创建进度条
        pbar = tqdm(total=len(self.video_paths), desc="预加载视频")
        
        # 使用线程池加载视频
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(load_worker, idx): idx for idx in range(len(self.video_paths))}
            
            # 处理完成的任务
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"处理视频 {idx} 时发生错误: {str(e)}")
        
        pbar.close()

    def _load_video(self, idx):
        """加载单个视频到内存"""
        if idx in self.cached_videos:
            return self.cached_videos[idx]
            
        video_path = self.video_paths[idx]
        try:
            logger.debug(f"处理视频: {video_path}")
            
            # 预处理视频
            processed_path = self.preprocess_video(video_path)
            if processed_path is None:
                frames_tensor = torch.zeros((self.max_frames, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                return frames_tensor, torch.tensor(-1)
            
            # 使用更安全的视频打开方式
            container = av.open(processed_path, options={
                'threads': '1',
                'skip_frame': 'nokey',
                'flags': 'low_delay',
                'strict': 'experimental'
            })
            
            frames = []
            
            # 获取视频信息
            stream = container.streams.video[0]
            total_frames = stream.frames
            
            # 使用更安全的帧采样方法
            if total_frames > 0:
                indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            else:
                indices = np.arange(0, self.max_frames * 2, 2)
            
            # 使用更安全的帧解码方式
            for i, frame in enumerate(container.decode(video=0)):
                try:
                    if i in indices:
                        frame_array = frame.to_ndarray(format="rgb24")
                        if frame_array is not None and frame_array.size > 0:
                            # 调整帧大小
                            frame_array = cv2.resize(frame_array, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                            frames.append(frame_array)
                    if len(frames) >= self.max_frames:
                        break
                except Exception as frame_error:
                    logger.warning(f"解码帧 {i} 时出错: {str(frame_error)}")
                    continue
                    
            container.close()
            
            # 清理临时文件
            try:
                os.unlink(processed_path)
            except:
                pass
            
            # 处理帧数不足的情况
            if len(frames) < self.max_frames:
                logger.debug(f"视频帧数不足 ({len(frames)} < {self.max_frames})，进行帧重复")
                if len(frames) > 0:
                    last_frame = frames[-1]
                    while len(frames) < self.max_frames:
                        frames.append(last_frame)
                else:
                    blank_frame = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
                    frames = [blank_frame] * self.max_frames
            
            # 处理视频帧
            logger.debug("提取视频特征")
            # 将帧列表转换为numpy数组
            frames_array = np.stack(frames)
            # 转换为张量并调整维度顺序
            frames_tensor = torch.from_numpy(frames_array).float()
            # 调整维度顺序为 (num_frames, channels, height, width)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)
            # 归一化
            frames_tensor = frames_tensor / 255.0
            
            return frames_tensor, torch.tensor(self.labels[idx])
            
        except Exception as e:
            logger.error(f"处理视频时出错 {video_path}: {str(e)}")
            logger.debug(f"错误详情: {str(e)}", exc_info=True)
            frames_tensor = torch.zeros((self.max_frames, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            return frames_tensor, torch.tensor(-1)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if self.preload:
            return self.cached_videos[idx]
        else:
            return self._load_video(idx)

    def to_device(self, device):
        """将预加载的视频转移到指定设备"""
        if not self.preload:
            logger.warning("数据集未预加载，无法转移到设备")
            return
            
        logger.info(f"开始将视频数据转移到设备: {device}")
        
        def transfer_worker(idx):
            try:
                frames_tensor, label = self.cached_videos[idx]
                with self.cache_lock:
                    self.cached_videos[idx] = (frames_tensor.to(device), label.to(device))
                return idx
            except Exception as e:
                logger.error(f"转移视频 {idx} 到设备失败: {str(e)}")
                return None

        # 创建进度条
        pbar = tqdm(total=len(self.video_paths), desc="转移数据到设备")
        
        # 使用线程池转移数据
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(transfer_worker, idx): idx for idx in range(len(self.video_paths))}
            
            # 处理完成的任务
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"处理视频 {idx} 时发生错误: {str(e)}")
        
        pbar.close()
        logger.info("数据转移完成")

    def preprocess_video(self, video_path):
        """预处理视频文件"""
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            command = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-y',
                temp_path
            ]
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"视频转码失败: {stderr.decode()}")
                return None
                
            return temp_path
            
        except Exception as e:
            logger.error(f"视频预处理失败: {str(e)}")
            return None

def prepare_dataset(folder_path):
    """准备数据集"""
    logger.info(f"开始准备数据集，数据目录: {folder_path}")
    video_paths = []
    labels = []
    
    # 处理fighting类别
    fighting_folder = os.path.join(folder_path, "Violence")
    fighting_videos = [f for f in os.listdir(fighting_folder) if f.endswith('.mp4')]
    logger.info(f"找到 {len(fighting_videos)} 个fighting视频")
    for video_file in fighting_videos:
        video_paths.append(os.path.join(fighting_folder, video_file))
        labels.append(1)  # fighting类别标签为1
    
    # 处理non_fighting类别
    non_fighting_folder = os.path.join(folder_path, "NonViolence")
    non_fighting_videos = [f for f in os.listdir(non_fighting_folder) if f.endswith('.mp4')]
    logger.info(f"找到 {len(non_fighting_videos)} 个non_fighting视频")
    for video_file in non_fighting_videos:
        video_paths.append(os.path.join(non_fighting_folder, video_file))
        labels.append(0)  # non_fighting类别标签为0
    
    logger.info(f"数据集准备完成，共 {len(video_paths)} 个视频")
    return video_paths, labels

def compute_metrics(eval_pred):
    """计算评估指标"""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    """数据批处理函数"""
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建保存目录
    save_dir = "./best_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载预训练模型和处理器
    logger.info("开始加载预训练模型...")
    try:
        processor = VideoMAEImageProcessor.from_pretrained(Config.MODEL_CKPT)
        model = VideoMAEForVideoClassification.from_pretrained(
            Config.MODEL_CKPT,
            num_labels=2,  # 二分类：fighting vs non-fighting
            ignore_mismatched_sizes=True
        )
        model = model.to(device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return
    
    # 准备数据集
    logger.info("开始准备数据集...")
    video_paths, labels = prepare_dataset("training_videos")
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42
    )
    logger.info(f"数据集划分完成: 训练集 {len(train_paths)} 个样本, 验证集 {len(val_paths)} 个样本")
    
    # 创建数据集并预加载（使用16个线程）
    train_dataset = VideoDataset(train_paths, train_labels, processor, preload=True, num_workers=16)
    val_dataset = VideoDataset(val_paths, val_labels, processor, preload=True, num_workers=16)
    
    # 将数据转移到GPU
    train_dataset.to_device(device)
    val_dataset.to_device(device)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        remove_unused_columns=False,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=10,
        # 禁用pin_memory，因为数据已经在GPU上
        dataloader_pin_memory=False,
        # 设置数据加载器的工作进程数
        dataloader_num_workers=0,  # 因为数据已经预加载，不需要额外的工作进程
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # 开始训练
    logger.info("开始训练模型...")
    trainer.train()
    
    # 保存最佳模型
    logger.info("保存最佳模型...")
    best_model_path = os.path.join(save_dir, "model")
    trainer.save_model(best_model_path)
    
    # 保存处理器
    processor.save_pretrained(best_model_path)
    
    # 保存训练配置
    config = {
        "model_name": Config.MODEL_CKPT,
        "num_labels": 2,
        "batch_size": Config.BATCH_SIZE,
        "max_frames": Config.MAX_FRAMES,
        "learning_rate": Config.LEARNING_RATE,
        "num_epochs": Config.NUM_EPOCHS
    }
    import json
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # 评估并保存指标
    logger.info("评估模型性能...")
    test_results = trainer.evaluate(val_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()
    
    logger.info(f"模型已保存到: {best_model_path}")
    logger.info(f"测试结果: {test_results}")

if __name__ == "__main__":
    main()


