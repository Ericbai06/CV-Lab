import kagglehub
import os
import json
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import av
import numpy as np
import pandas as pd
import requests
import zipfile
import kaggle

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_kinetics_labels():
    """加载Kinetics-400类别标签"""
    try:
        with open('kinetics400_labels.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("未找到kinetics400_labels.json文件")
        return None

def download_dataset():
    logger.info("开始下载数据集...")
    try:
        path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
        logger.info(f"数据集下载完成，路径: {path}")
        print("Path to dataset files:", path)
        return path
    except Exception as e:
        logger.error(f"数据集下载失败: {str(e)}")
        return None

def organize_dataset(dataset_path):
    logger.info("开始整理数据集...")
    os.makedirs('training_videos/Violence', exist_ok=True)
    os.makedirs('training_videos/NonViolence', exist_ok=True)
    violence_dir = os.path.join(dataset_path, 'Violence')
    non_violence_dir = os.path.join(dataset_path, 'NonViolence')
    if os.path.exists(violence_dir):
        for video in tqdm(os.listdir(violence_dir), desc="处理暴力视频"):
            src = os.path.join(violence_dir, video)
            dst = os.path.join('training_videos/Violence', video)
            shutil.copy2(src, dst)
    if os.path.exists(non_violence_dir):
        for video in tqdm(os.listdir(non_violence_dir), desc="处理非暴力视频"):
            src = os.path.join(non_violence_dir, video)
            dst = os.path.join('training_videos/NonViolence', video)
            shutil.copy2(src, dst)
    logger.info("数据集整理完成")

def process_video(video_path, output_dir, scene_duration=5.0, overlap=1.0):
    """
    处理视频，提取固定时长的场景
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        scene_duration: 场景持续时间（秒）
        overlap: 场景重叠时间（秒）
    """
    try:
        # 打开视频
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = stream.duration * stream.time_base
        
        # 计算场景数量
        num_scenes = int(duration / (scene_duration - overlap))
        
        # 提取场景
        for i in range(num_scenes):
            start_time = i * (scene_duration - overlap)
            end_time = start_time + scene_duration
            
            # 生成输出文件名
            output_name = f"{Path(video_path).stem}_{i:04d}.mp4"
            output_path = Path(output_dir) / output_name
            
            # 提取场景
            if extract_scene(video_path, start_time, end_time, output_path):
                logger.info(f"提取场景: {output_name}")
            
        container.close()
        return True
        
    except Exception as e:
        logger.error(f"处理视频失败: {str(e)}")
        return False

def extract_scene(video_path, start_time, end_time, output_path):
    """
    从视频中提取场景
    Args:
        video_path: 输入视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出路径
    """
    try:
        # 打开输入视频
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # 计算帧索引
        start_frame = int(start_time * stream.rate)
        end_frame = int(end_time * stream.rate)
        
        # 创建输出容器
        output_container = av.open(str(output_path), mode='w')
        output_stream = output_container.add_stream('h264', rate=stream.rate)
        output_stream.width = stream.width
        output_stream.height = stream.height
        
        # 提取帧
        container.seek(start_frame)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_frame - start_frame:
                break
            packet = output_stream.encode(frame)
            output_container.mux(packet)
            
        output_container.close()
        container.close()
        return True
        
    except Exception as e:
        logger.error(f"提取场景失败: {str(e)}")
        return False

def prepare_dataset():
    """准备数据集"""
    # 下载数据集
    dataset_path = download_dataset()
    if dataset_path is None:
        return False
    
    # 整理数据集
    organize_dataset(dataset_path)
    
    # 处理视频
    training_dir = Path("training_videos")
    
    # 处理打斗视频
    fighting_dir = training_dir / "Violence"
    for video_file in tqdm(list(fighting_dir.glob("*.mp4")), desc="处理打斗视频"):
        process_video(
            video_file,
            fighting_dir,
            scene_duration=5.0,
            overlap=1.0
        )
    
    # 处理非打斗视频
    non_fighting_dir = training_dir / "NonViolence"
    for video_file in tqdm(list(non_fighting_dir.glob("*.mp4")), desc="处理非打斗视频"):
        process_video(
            video_file,
            non_fighting_dir,
            scene_duration=5.0,
            overlap=1.0
        )
    
    return True

def main():
    """主函数"""
    try:
        if prepare_dataset():
            logger.info("数据集准备完成！")
        else:
            logger.error("数据集准备失败！")
    except Exception as e:
        logger.error(f"准备数据集时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 