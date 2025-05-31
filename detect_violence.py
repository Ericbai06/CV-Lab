import os
import torch
import cv2
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from transformers import VideoMAEForVideoClassification
import logging
from train_violence_classifier import ViolenceDetector, LongVideoDataset
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('violence_detection.log'),
        logging.StreamHandler()
    ]
)

# 设置环境变量来配置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def calculate_average_video_length(dataset_path):
    """计算数据集中视频的平均长度，并返回最接近的16的倍数"""
    total_frames = 0
    video_count = 0
    
    # 遍历fight和normal目录
    for video_type in ['fight', 'normal']:
        video_dir = os.path.join(dataset_path, 'videos', video_type)
        if not os.path.exists(video_dir):
            continue
            
        for video_file in os.listdir(video_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frames += frame_count
                    video_count += 1
                cap.release()
    
    if video_count == 0:
        return 16  # 默认值
    
    average_frames = total_frames / video_count
    # 计算最接近的16的倍数
    multiple = round(average_frames / 16)
    window_size = multiple * 16
    
    print(f"原始平均帧数: {average_frames:.2f}")
    print(f"最接近的16的倍数: {window_size}")
    
    return window_size

# 计算平均视频长度并设置滑动窗口大小
DATASET_PATH = "/root/autodl-tmp/UBI_FIGHTS"
FRAME_BATCH_SIZE = calculate_average_video_length(DATASET_PATH)
print(f"设置滑动窗口大小为: {FRAME_BATCH_SIZE} 帧")

# 修改预测参数
CONFIDENCE_THRESHOLD = 0.9  # 提高置信度阈值
MIN_SCENE_DURATION = 1.0  # 最小场景持续时间（秒）
MAX_SCENE_DURATION = 10.0  # 最大场景持续时间（秒）
FEATURE_BUFFER_SIZE = 32  # 特征缓冲区大小
MERGE_THRESHOLD = 2.0  # 合并相近检测的时间阈值（秒）

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.start_time = time.time()
        
    def update(self, frames_processed):
        current_time = time.time()
        elapsed = current_time - self.start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.memory_history.append(memory_used)
        
        self.start_time = current_time
        
    def get_stats(self):
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_memory = np.mean(self.memory_history) if self.memory_history else 0
        return {
            'fps': avg_fps,
            'memory_mb': avg_memory
        }

def load_model():
    """加载预训练模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = ViolenceDetector()
    model_path = "violence_classifier_checkpoint.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def process_video_frames(video_path, model, device):
    """逐帧处理视频"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化帧缓冲区和性能监控
    frame_buffer = []
    frame_times = []
    violence_probs = []
    timestamps = []  # 添加时间戳预测
    perf_monitor = PerformanceMonitor()
    
    # 设置滑动窗口参数
    window_size = 16  # VideoMAE模型期望的帧数
    stride = 4  # 减小滑动步长，提高检测精度
    
    # 初始化窗口预测结果列表
    window_predictions = []
    
    pbar = tqdm(total=total_frames, desc=f"处理视频 {os.path.basename(video_path)}", unit="frame")
    frames_processed = 0
    
    # 添加调试信息
    max_confidence = 0.0
    threshold_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理当前帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame_buffer.append(frame)
        frame_times.append(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)
        
        # 当缓冲区达到窗口大小时进行预测
        if len(frame_buffer) >= window_size:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            window_start_frame = current_frame - window_size
            
            # 转换为张量并归一化
            frames_tensor = torch.tensor(np.array(frame_buffer[:window_size]), dtype=torch.float32)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2) / 255.0
            frames_tensor = frames_tensor.unsqueeze(0)
            
            # 将数据移到正确的设备上
            frames_tensor = frames_tensor.to(device)
            
            with torch.no_grad():
                classification, pred_timestamps = model(frames_tensor)
                # 使用sigmoid获取置信度
                confidence = torch.sigmoid(classification[:, 1]).cpu().numpy()[0]
                violence_probs.append(confidence)  # 直接使用sigmoid输出作为概率
                
                # 保存窗口预测结果
                window_pred = {
                    'start_frame': window_start_frame,
                    'end_frame': current_frame,
                    'start_time': frame_times[-window_size],
                    'end_time': frame_times[-1],
                    'confidence': confidence,
                    'is_violence': confidence > CONFIDENCE_THRESHOLD
                }
                window_predictions.append(window_pred)
                
                # 更新调试信息
                max_confidence = max(max_confidence, confidence)
                if confidence > CONFIDENCE_THRESHOLD:
                    threshold_count += 1
                
                # 保存时间戳预测
                if confidence > CONFIDENCE_THRESHOLD:
                    pred_timestamps = torch.sigmoid(pred_timestamps)  # 将预测值转换到[0,1]范围
                    # 将相对帧号转换为全局帧号
                    rel_start_frame = int(pred_timestamps[0, 0].item() * window_size)
                    rel_end_frame = int(pred_timestamps[0, 1].item() * window_size)
                    global_start_frame = window_start_frame + rel_start_frame
                    global_end_frame = window_start_frame + rel_end_frame
                    timestamps.append((global_start_frame, global_end_frame))
                    # 添加调试信息
                    print(f"\n检测到潜在暴力场景:")
                    print(f"时间: {format_time(frame_times[-window_size])} - {format_time(frame_times[-1])}")
                    print(f"置信度: {confidence:.4f}")
                    print(f"窗口起始帧: {window_start_frame}")
                    print(f"相对预测片段: {rel_start_frame} - {rel_end_frame}")
                    print(f"全局预测片段: {global_start_frame} - {global_end_frame}")
            
            # 更新性能统计
            frames_processed += stride
            perf_monitor.update(frames_processed)
            stats = perf_monitor.get_stats()
            
            # 更新进度条
            pbar.set_postfix({
                'FPS': f'{stats["fps"]:.1f}',
                'Memory': f'{stats["memory_mb"]:.1f}MB',
                'Max Conf': f'{max_confidence:.4f}',
                'Threshold Count': threshold_count
            })
            
            # 滑动窗口
            frame_buffer = frame_buffer[stride:]
        
        pbar.update(1)
    
    # 处理剩余的帧
    if frame_buffer:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        window_start_frame = current_frame - len(frame_buffer)
        
        # 如果剩余帧数不足，重复最后一帧
        if len(frame_buffer) < window_size:
            last_frame = frame_buffer[-1]
            frame_buffer.extend([last_frame] * (window_size - len(frame_buffer)))
        
        frames_tensor = torch.tensor(np.array(frame_buffer[:window_size]), dtype=torch.float32)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2) / 255.0
        frames_tensor = frames_tensor.unsqueeze(0)
        
        # 将数据移到正确的设备上
        frames_tensor = frames_tensor.to(device)
        
        with torch.no_grad():
            classification, pred_timestamps = model(frames_tensor)
            # 使用sigmoid获取置信度
            confidence = torch.sigmoid(classification[:, 1]).cpu().numpy()[0]
            violence_probs.append(confidence)  # 直接使用sigmoid输出作为概率
            
            # 保存窗口预测结果
            window_pred = {
                'start_frame': window_start_frame,
                'end_frame': current_frame,
                'start_time': frame_times[-len(frame_buffer)],
                'end_time': frame_times[-1],
                'confidence': confidence,
                'is_violence': confidence > CONFIDENCE_THRESHOLD
            }
            window_predictions.append(window_pred)
            
            # 更新调试信息
            max_confidence = max(max_confidence, confidence)
            if confidence > CONFIDENCE_THRESHOLD:
                threshold_count += 1
            
            if confidence > CONFIDENCE_THRESHOLD:
                pred_timestamps = torch.sigmoid(pred_timestamps)
                # 将相对帧号转换为全局帧号
                rel_start_frame = int(pred_timestamps[0, 0].item() * window_size)
                rel_end_frame = int(pred_timestamps[0, 1].item() * window_size)
                global_start_frame = window_start_frame + rel_start_frame
                global_end_frame = window_start_frame + rel_end_frame
                timestamps.append((global_start_frame, global_end_frame))
                # 添加调试信息
                print(f"\n检测到潜在暴力场景:")
                print(f"时间: {format_time(frame_times[-len(frame_buffer)])} - {format_time(frame_times[-1])}")
                print(f"置信度: {confidence:.4f}")
                print(f"窗口起始帧: {window_start_frame}")
                print(f"相对预测片段: {rel_start_frame} - {rel_end_frame}")
                print(f"全局预测片段: {global_start_frame} - {global_end_frame}")
    
    cap.release()
    pbar.close()
    
    # 保存窗口预测结果到CSV文件
    df = pd.DataFrame(window_predictions)
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{timestamp}_windows.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n窗口预测结果已保存到: {csv_path}")
    
    # 打印总体统计信息
    print(f"\n视频处理统计:")
    print(f"最大置信度: {max_confidence:.4f}")
    print(f"超过阈值的检测次数: {threshold_count}")
    print(f"总帧数: {total_frames}")
    print(f"总窗口数: {len(window_predictions)}")
    
    return frame_times, violence_probs, timestamps

def merge_detections(detections, merge_threshold=MERGE_THRESHOLD):
    """
    合并相近的检测结果
    Args:
        detections: 检测结果列表，每个元素为(start_time, end_time, confidence)
        merge_threshold: 合并阈值（秒）
    Returns:
        合并后的检测结果列表
    """
    if not detections:
        return []
    
    # 按开始时间排序
    detections = sorted(detections, key=lambda x: x[0])
    merged = []
    current = list(detections[0])
    
    for next_det in detections[1:]:
        # 如果当前检测与下一个检测的时间间隔小于阈值
        if next_det[0] - current[1] <= merge_threshold:
            # 更新结束时间和置信度
            current[1] = max(current[1], next_det[1])
            current[2] = max(current[2], next_det[2])
        else:
            # 保存当前检测结果
            merged.append(tuple(current))
            current = list(next_det)
    
    # 添加最后一个检测结果
    merged.append(tuple(current))
    return merged

def detect_violence_scenes(frame_times, violence_probs, timestamps):
    """检测暴力场景"""
    scenes = []
    
    # 使用模型预测的时间戳
    for i, (start_frame, end_frame) in enumerate(timestamps):
        start_time = frame_times[start_frame]
        end_time = frame_times[min(end_frame, len(frame_times)-1)]
        duration = end_time - start_time
        
        # 只保留符合时长要求的场景
        if MIN_SCENE_DURATION <= duration <= MAX_SCENE_DURATION:
            # 计算该时间段内的平均暴力概率
            scene_probs = violence_probs[start_frame:end_frame+1]
            avg_prob = np.mean(scene_probs) if scene_probs else 0
            
            if avg_prob > CONFIDENCE_THRESHOLD:
                scenes.append((start_time, end_time, avg_prob))
    
    # 合并相近的检测结果
    merged_scenes = merge_detections(scenes)
    
    # 打印检测统计信息
    print(f"\n检测统计:")
    print(f"原始检测数: {len(scenes)}")
    print(f"合并后检测数: {len(merged_scenes)}")
    
    return merged_scenes

def save_results_to_json(video_path, scenes, output_dir="detection_results"):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def format_time_for_json(seconds):
        """将秒数转换为时分秒格式的字符串"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    result = {
        "video_name": video_name,
        "detection_time": timestamp,
        "total_scenes": len(scenes),
        "scenes": [
            {
                "scene_id": i + 1,
                "start_time": format_time_for_json(start_time),
                "end_time": format_time_for_json(end_time),
                "duration": format_time_for_json(end_time - start_time),
                "confidence": prob
            }
            for i, (start_time, end_time, prob) in enumerate(scenes)
        ]
    }
    
    output_file = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"\n检测结果已保存到: {output_file}")
    return output_file

def format_time(seconds):
    """将秒数转换为时分秒格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def main():
    # 要处理的视频列表
    video_paths = [
        "1.mp4",
        "2.mp4"
    ]
    
    # 逐个处理视频
    for video_path in video_paths:
        print(f"\n开始处理视频: {video_path}")
        model, device = load_model()
        
        # 逐帧处理视频
        frame_times, violence_probs, timestamps = process_video_frames(video_path, model, device)
        
        # 检测暴力场景
        violence_scenes = detect_violence_scenes(frame_times, violence_probs, timestamps)
        
        if violence_scenes:
            print(f"\n{video_path} 检测到的暴力场景:")
            for i, (start_time, end_time, prob) in enumerate(violence_scenes):
                print(f"场景 {i+1}:")
                print(f"时间段: {format_time(start_time)} - {format_time(end_time)}")
                print(f"持续时间: {format_time(end_time - start_time)}")
                print(f"暴力概率: {prob:.2%}")
                print("---")
            
            save_results_to_json(video_path, violence_scenes)
        else:
            print(f"{video_path} 未检测到暴力场景")
            save_results_to_json(video_path, [])
        
        print(f"\n{video_path} 处理完成")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 