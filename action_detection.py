#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打斗场景检测模型
使用MobileNetV3和ViViT架构实现的视频打斗场景检测
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import VivitImageProcessor, VivitForVideoClassification
import json
from datetime import datetime
import psutil
import gc

# 超参数设置
BATCH_SIZE = 8
MAX_FRAMES = 32  # ViViT默认帧数
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
DATA_NUM = 200
PROJECTION_DIM = 128  # 特征维度

# 模型保存参数
MODEL_SAVE_DIR = "saved_models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
HISTORY_PATH = os.path.join(MODEL_SAVE_DIR, "training_history.json")

class MobileNetV3FeatureExtractor(nn.Module):
    """MobileNetV3特征提取器"""
    def __init__(self, trainable=True):
        super().__init__()
        
        # 加载预训练的MobileNetV3模型
        self.mobilenetv3 = mobilenet_v3_small(pretrained=True)
        
        # 移除最后的分类层
        self.mobilenetv3 = nn.Sequential(*list(self.mobilenetv3.children())[:-1])
        
        # 设置是否可训练
        for param in self.mobilenetv3.parameters():
            param.requires_grad = trainable
            
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(576, PROJECTION_DIM)  # 576是MobileNetV3-Small的特征维度

    def forward(self, x):
        # 提取特征
        features = self.mobilenetv3(x)
        features = self.pooling(features)
        features = features.view(features.size(0), -1)
        features = self.dense(features)
        return features

class VideoDataset(Dataset):
    """视频数据集类"""
    def __init__(self, video_paths, labels, max_frames=MAX_FRAMES, resize=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.max_frames = max_frames
        self.resize = resize

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # 加载视频
        frames = load_video(video_path, self.max_frames, self.resize)
        
        # 转换为PyTorch张量
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        return frames, label

def load_video(path, max_frames=MAX_FRAMES, resize=(224, 224)):
    """
    加载视频文件并提取帧
    
    Args:
        path: 视频文件路径
        max_frames: 最大帧数
        resize: 调整大小
    
    Returns:
        numpy.ndarray: 视频帧数组
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            frame_count += 1
    finally:
        cap.release()

    # 用最后一帧填充不足的帧
    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)

def train_model(model, feature_extractor, train_loader, val_loader, device):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': feature_extractor.parameters()}
    ], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5, min_lr=5e-5
    )
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        feature_extractor.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for frames, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            frames, labels = frames.to(device), labels.to(device)
            
            # 提取特征
            batch_size, num_frames, c, h, w = frames.size()
            frames = frames.view(batch_size * num_frames, c, h, w)
            features = feature_extractor(frames)
            features = features.view(batch_size, num_frames, -1)
            
            optimizer.zero_grad()
            outputs = model(inputs_embeds=features)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        feature_extractor.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                
                # 提取特征
                batch_size, num_frames, c, h, w = frames.size()
                frames = frames.view(batch_size * num_frames, c, h, w)
                features = feature_extractor(frames)
                features = features.view(batch_size, num_frames, -1)
                
                outputs = model(inputs_embeds=features)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, BEST_MODEL_PATH)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history

def main():
    """主函数"""
    # 创建保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    print("加载预训练模型...")
    model_name = "google/vivit-b-16x2-kinetics400"
    model = VivitForVideoClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # 创建特征提取器
    feature_extractor = MobileNetV3FeatureExtractor().to(device)
    
    # 准备数据集
    print("准备数据集...")
    dataset_path = "/root/autodl-tmp/Cv/training_videos"
    class_names = ["NonViolence", "Violence"]
    video_paths = []
    labels = []
    
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        video_cnt = 0
        for video_file in os.listdir(class_folder):
            if video_cnt > DATA_NUM:
                break
            video_path = os.path.join(class_folder, video_file)
            video_paths.append(video_path)
            labels.append(class_index)
            video_cnt += 1
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.1, random_state=123
    )
    
    # 创建数据集和数据加载器
    train_dataset = VideoDataset(train_paths, train_labels)
    val_dataset = VideoDataset(val_paths, val_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # 训练模型
    print("开始训练...")
    history = train_model(model, feature_extractor, train_loader, val_loader, device)
    
    # 保存训练历史
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history.png'))
    plt.close()
    
    print(f"\n训练完成！")
    print(f"最佳模型已保存至: {BEST_MODEL_PATH}")
    print(f"训练历史已保存至: {HISTORY_PATH}")

if __name__ == "__main__":
    main() 