import os
# 设置环境变量，禁用不必要的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用 TensorFlow 日志
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步 CUDA 操作

import torch
import numpy as np
import cv2
from transformers import VivitImageProcessor, VivitForVideoClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import av
import logging
import sys
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
    logger.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 配置参数
class Config:
    BATCH_SIZE = 16  # 增加批次大小
    MAX_FRAMES = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2  # 减少工作进程数
    PREFETCH_FACTOR = 2
    MIXED_PRECISION = True

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor, model, max_frames=Config.MAX_FRAMES):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.model = model
        self.max_frames = max_frames
        logger.info(f"初始化数据集，共 {len(video_paths)} 个视频")

    def __len__(self):
        return len(self.video_paths)

    def preprocess_video(self, video_path):
        """预处理视频文件"""
        try:
            # 使用ffmpeg进行预处理
            import subprocess
            import tempfile
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 使用ffmpeg进行转码
            command = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',  # 使用x264编码器
                '-preset', 'ultrafast',  # 使用最快的编码速度
                '-tune', 'zerolatency',  # 优化低延迟
                '-profile:v', 'baseline',  # 使用基准配置文件
                '-level', '3.0',  # 设置兼容级别
                '-pix_fmt', 'yuv420p',  # 使用标准像素格式
                '-y',  # 覆盖输出文件
                temp_path
            ]
            
            # 执行转码
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

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            logger.debug(f"处理视频: {video_path}")
            
            # 预处理视频
            processed_path = self.preprocess_video(video_path)
            if processed_path is None:
                return torch.zeros(768).to(device), torch.tensor(-1)
            
            # 使用更安全的视频打开方式
            container = av.open(processed_path, options={
                'threads': '1',
                'skip_frame': 'nokey',  # 只解码关键帧
                'flags': 'low_delay',  # 低延迟模式
                'strict': 'experimental'  # 使用实验性解码器
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
                            frame_array = cv2.resize(frame_array, (96, 96))
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
                    # 如果没有任何有效帧，创建空白帧
                    blank_frame = np.zeros((96, 96, 3), dtype=np.uint8)
                    frames = [blank_frame] * self.max_frames
            
            # 处理视频帧
            logger.debug("提取视频特征")
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                features = outputs.hidden_states[-1].mean(dim=1)
            
            return features.squeeze(), torch.tensor(label)
            
        except Exception as e:
            logger.error(f"处理视频时出错 {video_path}: {str(e)}")
            logger.debug(f"错误详情: {str(e)}", exc_info=True)
            return torch.zeros(768).to(device), torch.tensor(-1)

def prepare_dataset(folder_path):
    """准备数据集"""
    logger.info(f"开始准备数据集，数据目录: {folder_path}")
    video_paths = []
    labels = []
    
    # 处理fighting类别
    fighting_folder = os.path.join(folder_path, "fighting")
    fighting_videos = [f for f in os.listdir(fighting_folder) if f.endswith('.mp4')]
    logger.info(f"找到 {len(fighting_videos)} 个fighting视频")
    for video_file in fighting_videos:
        video_paths.append(os.path.join(fighting_folder, video_file))
        labels.append(1)  # fighting类别标签为1
    
    # 处理non_fighting类别
    non_fighting_folder = os.path.join(folder_path, "non_fighting")
    non_fighting_videos = [f for f in os.listdir(non_fighting_folder) if f.endswith('.mp4')]
    logger.info(f"找到 {len(non_fighting_videos)} 个non_fighting视频")
    for video_file in non_fighting_videos:
        video_paths.append(os.path.join(non_fighting_folder, video_file))
        labels.append(0)  # non_fighting类别标签为0
    
    logger.info(f"数据集准备完成，共 {len(video_paths)} 个视频")
    return video_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=Config.NUM_EPOCHS):
    """训练模型"""
    logger.info(f"使用设备: {device}")
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()  # 用于混合精度训练
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"\n开始第 {epoch+1}/{num_epochs} 轮训练")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        invalid_samples = 0
        
        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # 过滤掉无效样本
            valid_mask = labels != -1
            if not valid_mask.any():
                invalid_samples += 1
                continue
                
            # 将数据移到GPU
            features = features[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if Config.MIXED_PRECISION:
                with autocast("cuda", dtype=torch.float16):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                if torch.cuda.is_available():
                    logger.debug(f"GPU显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_invalid_samples = 0
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                # 过滤掉无效样本
                valid_mask = labels != -1
                if not valid_mask.any():
                    val_invalid_samples += 1
                    continue
                    
                # 将数据移到GPU
                features = features[valid_mask].to(device)
                labels = labels[valid_mask].to(device)
                
                if Config.MIXED_PRECISION:
                    with autocast("cuda", dtype=torch.float16):
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        logger.info(f"\nEpoch {epoch+1}/{num_epochs} 训练结果:")
        logger.info(f"训练集: Loss = {train_loss/len(train_loader):.4f}, Acc = {train_acc:.2f}%")
        logger.info(f"验证集: Loss = {val_loss/len(val_loader):.4f}, Acc = {val_acc:.2f}%")
        logger.info(f"无效样本数: 训练集 {invalid_samples}, 验证集 {val_invalid_samples}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

def predict_video(model, video_path, processor, max_frames=32):
    """对单个视频进行预测"""
    model = model.to(device)
    model.eval()
    
    def preprocess_video(video_path):
        """预处理视频文件"""
        try:
            # 使用ffmpeg进行预处理
            import subprocess
            import tempfile
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 使用ffmpeg进行转码
            command = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',  # 使用x264编码器
                '-preset', 'ultrafast',  # 使用最快的编码速度
                '-tune', 'zerolatency',  # 优化低延迟
                '-profile:v', 'baseline',  # 使用基准配置文件
                '-level', '3.0',  # 设置兼容级别
                '-pix_fmt', 'yuv420p',  # 使用标准像素格式
                '-y',  # 覆盖输出文件
                temp_path
            ]
            
            # 执行转码
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
    
    try:
        logger.info(f"开始处理视频: {video_path}")
        
        # 预处理视频
        processed_path = preprocess_video(video_path)
        if processed_path is None:
            return None, None
        
        # 使用更安全的视频打开方式
        container = av.open(processed_path, options={
            'threads': '1',
            'skip_frame': 'nokey',  # 只解码关键帧
            'flags': 'low_delay',  # 低延迟模式
            'strict': 'experimental'  # 使用实验性解码器
        })
        
        frames = []
        
        # 获取视频信息
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        # 使用更安全的帧采样方法
        if total_frames > 0:
            indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            indices = np.arange(0, max_frames * 2, 2)
        
        # 使用更安全的帧解码方式
        for i, frame in enumerate(container.decode(video=0)):
            try:
                if i in indices:
                    frame_array = frame.to_ndarray(format="rgb24")
                    if frame_array is not None and frame_array.size > 0:
                        # 调整帧大小
                        frame_array = cv2.resize(frame_array, (96, 96))
                        frames.append(frame_array)
                if len(frames) >= max_frames:
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
        if len(frames) < max_frames:
            logger.debug(f"视频帧数不足 ({len(frames)} < {max_frames})，进行帧重复")
            if len(frames) > 0:
                last_frame = frames[-1]
                while len(frames) < max_frames:
                    frames.append(last_frame)
            else:
                # 如果没有任何有效帧，创建空白帧
                blank_frame = np.zeros((96, 96, 3), dtype=np.uint8)
                frames = [blank_frame] * max_frames
        
        # 处理视频帧
        logger.debug("提取视频特征")
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states[-1].mean(dim=1)
            features = features.to(device)
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"预测视频时出错 {video_path}: {str(e)}")
        logger.debug(f"错误详情: {str(e)}", exc_info=True)
        return None, None

def main():
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 设置模型路径
    model_path = os.path.abspath("vivit-b-16x2-kinetics400")
    logger.info(f"模型路径: {model_path}")
    
    # 加载预训练模型和处理器
    logger.info("开始加载预训练模型...")
    try:
        processor = VivitImageProcessor.from_pretrained(model_path, local_files_only=True)
        vivit_model = VivitForVideoClassification.from_pretrained(model_path, local_files_only=True)
        vivit_model = vivit_model.to(device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return
    
    # 创建分类模型
    logger.info("创建分类模型")
    classifier = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 2)
    ).to(device)
    
    # 准备数据集
    logger.info("开始准备数据集...")
    video_paths, labels = prepare_dataset("training_videos")
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42
    )
    logger.info(f"数据集划分完成: 训练集 {len(train_paths)} 个样本, 验证集 {len(val_paths)} 个样本")
    
    # 创建数据加载器
    train_dataset = VideoDataset(train_paths, train_labels, processor, vivit_model)
    val_dataset = VideoDataset(val_paths, val_labels, processor, vivit_model)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True  # 保持工作进程存活
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True  # 保持工作进程存活
    )
    
    # 训练模型
    logger.info("开始训练模型...")
    train_model(classifier, train_loader, val_loader)
    
    # 加载最佳模型
    logger.info("加载最佳模型...")
    classifier.load_state_dict(torch.load("best_model.pth"))
    
    # 预测测试视频
    test_video_path = "The.Knockout.S01E01.mp4"
    if os.path.exists(test_video_path):
        logger.info(f"开始预测视频: {test_video_path}")
        predicted_class, confidence = predict_video(classifier, test_video_path, processor)
        if predicted_class is not None:
            class_name = "Fighting" if predicted_class == 1 else "Non-fighting"
            logger.info(f"预测结果: {class_name}, 置信度: {confidence:.2f}")
    else:
        logger.error(f"测试视频不存在: {test_video_path}")

if __name__ == "__main__":
    main() 