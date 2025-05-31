import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
import torch
import torch.nn as nn
from transformers.models.videomae.image_processing_videomae import VideoMAEImageProcessor
from transformers.models.videomae.modeling_videomae import VideoMAEModel
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
import seaborn as sns
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 设置Hugging Face镜像和缓存目录


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 线性变换
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(out)

class ViolenceDataset(Dataset):
    def __init__(self, root_dir, processor, transform=None):
        self.root_dir = root_dir    
        self.processor = processor
        self.transform = transform
        self.videos = []
        self.labels = []
        self.frame_labels = []
        
        # 分别处理fight和normal目录
        for video_type in ['fight', 'normal']:
            video_dir = os.path.join(root_dir, 'videos', video_type)
            if not os.path.exists(video_dir):
                print(f"警告: 目录不存在 {video_dir}")
                continue
                
            for video_file in os.listdir(video_dir):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(video_dir, video_file)
                    # 获取视频文件名（不含扩展名）
                    video_name = os.path.splitext(video_file)[0]
                    annotation_file = os.path.join(root_dir, 'annotation', f'{video_name}.csv')
                    
                    if os.path.exists(annotation_file):
                        try:
                            # 读取标注文件，没有列名
                            frame_labels = pd.read_csv(annotation_file, header=None)[0].values
                            self.videos.append(video_path)
                            self.frame_labels.append(frame_labels)
                            # 根据目录判断标签
                            self.labels.append(1 if video_type == 'fight' else 0)
                        except Exception as e:
                            print(f"处理文件出错 {annotation_file}: {str(e)}")
        
        print(f"数据集总共加载了 {len(self.videos)} 个视频文件")
        print(f"其中打斗视频: {sum(self.labels)} 个")
        print(f"非打斗视频: {len(self.labels) - sum(self.labels)} 个")

    def __len__(self):
        return len(self.videos)

    def get_fight_segments(self, frame_labels):
        """
        从逐帧标签中提取打斗片段的起止帧
        Args:
            frame_labels: 逐帧标签数组，0表示非打斗，1表示打斗
        Returns:
            start_frame, end_frame: 打斗片段的起止帧
        """
        # 找到所有打斗帧的位置
        fight_frames = np.where(frame_labels == 1)[0]
        if len(fight_frames) == 0:
            return 0, 0  # 如果没有打斗帧，返回[0, 0]
        
        # 找到连续的打斗片段
        segments = []
        start = fight_frames[0]
        for i in range(1, len(fight_frames)):
            if fight_frames[i] - fight_frames[i-1] > 1:  # 如果不连续
                segments.append((start, fight_frames[i-1]))
                start = fight_frames[i]
        segments.append((start, fight_frames[-1]))  # 添加最后一个片段
        
        # 选择最长的打斗片段
        longest_segment = max(segments, key=lambda x: x[1] - x[0])
        return longest_segment[0], longest_segment[1]

    def __getitem__(self, idx):
        try:
            video_path = self.videos[idx]
            label = self.labels[idx]
            frame_label = self.frame_labels[idx]
            
            # 读取视频帧
            frames = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return None
                
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"视频帧数为0: {video_path}")
                return None
                
            # 计算采样间隔
            if total_frames > 16:
                interval = total_frames // 16
                frame_indices = [i * interval for i in range(16)]
            else:
                frame_indices = list(range(total_frames))
                frame_indices = frame_indices * (16 // len(frame_indices) + 1)
                frame_indices = frame_indices[:16]
            
            # 读取指定帧
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"无法读取帧 {frame_idx} 在视频: {video_path}")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            # 确保有足够的帧
            if len(frames) < 16:
                print(f"视频帧数不足: {video_path}, 实际帧数: {len(frames)}")
                return None
                
            # 使用processor处理帧
            try:
                inputs = self.processor(frames, return_tensors="pt")
            except Exception as e:
                print(f"处理帧时出错: {video_path}, 错误: {str(e)}")
                return None
            
            # 获取打斗片段的起止帧
            if label == 1:  # 如果是打斗视频
                start_frame, end_frame = self.get_fight_segments(frame_label)
                # 将全局帧号转换为窗口内的相对帧号
                start_frame = max(0, min(15, int(start_frame * 16 / total_frames)))
                end_frame = max(0, min(15, int(end_frame * 16 / total_frames)))
                frame_label = torch.tensor([start_frame, end_frame], dtype=torch.float32)
            else:  # 如果是非打斗视频
                frame_label = torch.tensor([0, 0], dtype=torch.float32)
            
            return inputs, torch.tensor(label), frame_label
            
        except Exception as e:
            print(f"处理视频时出错: {video_path}, 错误: {str(e)}")
            return None

def collate_fn(batch):
    """
    处理批次数据
    Args:
        batch: 批次数据列表，每个元素为(inputs, label, frame_label)
    Returns:
        processed_inputs: 处理后的输入数据
        labels: 标签张量
        frame_labels: 帧标签张量
    """
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None
    
    # 分离输入和标签
    inputs, labels, frame_labels = zip(*batch)
    
    # 处理输入数据
    processed_inputs = {
        "pixel_values": torch.stack([item["pixel_values"].squeeze(0) for item in inputs])
    }
    
    # 处理标签
    labels = torch.stack(labels)
    frame_labels = torch.stack(frame_labels)
    
    return processed_inputs, labels, frame_labels

class ViolenceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 保持原有的VideoMAE特征提取器
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 时序特征提取层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 分类头 - 预测是否包含打斗
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类：打斗/非打斗
        )
        
        # 时序预测头 - 预测打斗片段的起止帧
        self.temporal_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 预测起止帧
        )
    
    def forward(self, pixel_values):
        # 获取视频特征
        features = self.videomae(pixel_values=pixel_values).last_hidden_state
        
        # 特征转换
        batch_size, seq_len, hidden_size = features.shape
        features = features.reshape(-1, hidden_size)
        features = self.feature_transform(features)
        features = features.reshape(batch_size, seq_len, -1)
        
        # 时序特征提取
        temporal_features = self.temporal_conv(features.transpose(1, 2))
        temporal_features = temporal_features.transpose(1, 2)
        
        # 视频分类 - 是否包含打斗
        video_features = temporal_features.mean(dim=1)  # 全局池化
        classification = self.classifier(video_features)
        
        # 时序预测 - 打斗片段的起止帧
        timestamps = self.temporal_predictor(video_features)  # 使用全局特征预测起止帧
        
        return classification, timestamps

def calculate_metrics(y_true, y_pred):
    """计算各种评估指标"""
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix, epoch):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
    plt.close()

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs):
    """
    绘制训练指标的变化趋势图
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
    """
    plt.figure(figsize=(15, 10))
    
    # 创建两个子图
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Training Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def temporal_loss(pred_timestamps, gt_timestamps):
    """
    计算时间戳预测损失，使用IoU类似的方法
    Args:
        pred_timestamps: 预测的时间戳 [batch_size, 2]
        gt_timestamps: 真实的时间戳 [batch_size, 2]
    Returns:
        loss: 基于IoU的损失值
    """
    # 归一化时间戳到[0,1]范围
    pred_timestamps = torch.sigmoid(pred_timestamps)
    
    # 计算交集
    intersection_start = torch.max(pred_timestamps[:, 0], gt_timestamps[:, 0])
    intersection_end = torch.min(pred_timestamps[:, 1], gt_timestamps[:, 1])
    intersection = torch.clamp(intersection_end - intersection_start, min=0)
    
    # 计算并集
    union_start = torch.min(pred_timestamps[:, 0], gt_timestamps[:, 0])
    union_end = torch.max(pred_timestamps[:, 1], gt_timestamps[:, 1])
    union = union_end - union_start
    
    # 计算IoU
    iou = intersection / (union + 1e-6)
    
    # 将IoU转换为损失（1 - IoU）
    iou_loss = 1 - iou
    
    # 时间连续性损失 - 确保结束时间大于开始时间
    time_diff = pred_timestamps[:, 1] - pred_timestamps[:, 0]
    continuity_loss = torch.mean(F.softplus(-time_diff))
    
    # 总损失
    total_loss = torch.mean(iou_loss) + 0.05 * continuity_loss
    
    return total_loss

def train():
    # 初始化处理器和模型
    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base",
        resume_download=True
    )
    
    # 加载预训练模型
    model = ViolenceDetector()
    model.videomae = VideoMAEModel.from_pretrained(
        "MCG-NJU/videomae-base",
        ignore_mismatched_sizes=True,
        resume_download=True
    )
    
    # 准备数据集
    root_dir = "/root/autodl-tmp/UBI_FIGHTS"
    dataset = ViolenceDataset(root_dir, processor=processor)
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    # 训练设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 使用较小的学习率进行微调
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # 定义checkpoint保存路径
    checkpoint_path = "violence_classifier_checkpoint.pth"
    best_model_path = "best_violence_classifier.pth"

    # 初始化早停
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    # 训练循环
    num_epochs = 10
    best_val_loss = float('inf')
    
    # 初始化指标列表
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_classification_loss = 0
        total_temporal_loss = 0
        total_iou = 0  # 添加IoU统计
        correct = 0
        total = 0
        
        # 添加loss平滑
        classification_loss_smooth = 0
        temporal_loss_smooth = 0
        iou_smooth = 0  # 添加IoU平滑
        smooth_factor = 0.9
        
        for batch_idx, (inputs, labels, frame_labels) in enumerate(tqdm(train_dataloader)):
            if inputs is None or labels is None or frame_labels is None:
                continue
                
            pixel_values = inputs["pixel_values"].squeeze(1).to(device)
            labels = labels.to(device)
            frame_labels = frame_labels.to(device)
            
            optimizer.zero_grad()
            classification, timestamps = model(pixel_values)
            
            # 计算分类损失
            classification_loss = F.cross_entropy(classification, labels)
            
            # 计算时间戳损失 - 只对打斗视频计算
            fight_mask = (labels == 1)
            if fight_mask.any():
                temporal_loss_val = temporal_loss(
                    timestamps[fight_mask],
                    frame_labels[fight_mask]
                )
                
                # 计算IoU
                pred_timestamps = torch.sigmoid(timestamps[fight_mask])
                gt_timestamps = frame_labels[fight_mask]
                
                # 计算交集
                intersection_start = torch.max(pred_timestamps[:, 0], gt_timestamps[:, 0])
                intersection_end = torch.min(pred_timestamps[:, 1], gt_timestamps[:, 1])
                intersection = torch.clamp(intersection_end - intersection_start, min=0)
                
                # 计算并集
                union_start = torch.min(pred_timestamps[:, 0], gt_timestamps[:, 0])
                union_end = torch.max(pred_timestamps[:, 1], gt_timestamps[:, 1])
                union = union_end - union_start
                
                # 计算IoU
                iou = intersection / (union + 1e-6)
                batch_iou = torch.mean(iou).item()
            else:
                temporal_loss_val = torch.tensor(0.0, device=device)
                batch_iou = 0.0
            
            # 平滑loss和IoU
            if batch_idx == 0:
                classification_loss_smooth = classification_loss.item()
                temporal_loss_smooth = temporal_loss_val.item()
                iou_smooth = batch_iou
            else:
                classification_loss_smooth = smooth_factor * classification_loss_smooth + (1 - smooth_factor) * classification_loss.item()
                temporal_loss_smooth = smooth_factor * temporal_loss_smooth + (1 - smooth_factor) * temporal_loss_val.item()
                iou_smooth = smooth_factor * iou_smooth + (1 - smooth_factor) * batch_iou
            
            # 总损失
            loss = classification_loss + temporal_loss_val
            
            # 确保loss是标量
            if not isinstance(loss, torch.Tensor) or loss.dim() > 0:
                loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_temporal_loss += temporal_loss_val.item()
            total_iou += batch_iou
            
            # 计算准确率
            pred = classification.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx}")
                print(f"Total Loss: {loss.item():.4f}")
                print(f"Classification Loss: {classification_loss_smooth:.4f}")
                print(f"Temporal Loss: {temporal_loss_smooth:.4f}")
                print(f"IoU: {iou_smooth:.4f}")
                print(f"Accuracy: {100. * correct / total:.2f}%")
        
        avg_loss = total_loss / len(train_dataloader)
        avg_classification_loss = total_classification_loss / len(train_dataloader)
        avg_temporal_loss = total_temporal_loss / len(train_dataloader)
        avg_iou = total_iou / len(train_dataloader)
        train_acc = 100. * correct / total
        train_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Classification Loss: {avg_classification_loss:.4f}")
        print(f"Temporal Loss: {avg_temporal_loss:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_classification_loss = 0
        val_temporal_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, frame_labels in tqdm(val_dataloader, desc="Validating"):
                if inputs is None or labels is None or frame_labels is None:
                    continue
                    
                pixel_values = inputs["pixel_values"].squeeze(1).to(device)
                labels = labels.to(device)
                frame_labels = frame_labels.to(device)
                
                classification, timestamps = model(pixel_values)
                
                # 计算验证损失
                classification_loss = F.cross_entropy(classification, labels)
                
                # 计算时间戳损失 - 只对打斗视频计算
                fight_mask = (labels == 1)
                if fight_mask.any():
                    temporal_loss_val = temporal_loss(
                        timestamps[fight_mask],
                        frame_labels[fight_mask]
                    )
                else:
                    temporal_loss_val = torch.tensor(0.0, device=device)
                
                loss = classification_loss + 0.05 * temporal_loss_val
                
                val_loss += loss.item()
                val_classification_loss += classification_loss.item()
                val_temporal_loss += temporal_loss_val.item()
                
                pred = classification.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_classification_loss = val_classification_loss / len(val_dataloader)
        avg_val_temporal_loss = val_temporal_loss / len(val_dataloader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Classification Loss: {avg_val_classification_loss:.4f}")
        print(f"Validation Temporal Loss: {avg_val_temporal_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 检查早停
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # 保存checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")

    # 训练结束后绘制指标图
    plot_training_metrics(train_losses, val_losses, [], [])  # 只绘制损失曲线

class LongVideoDataset(Dataset):
    def __init__(self, video_path, window_size=16, stride=8):
        self.video_path = video_path
        self.window_size = window_size
        self.stride = stride
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        
        # 获取视频总帧数
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 计算可能的窗口数量
        self.num_windows = (self.total_frames - self.window_size) // self.stride + 1
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        start_frame = idx * self.stride
        frames = []
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.window_size):
            ret, frame = cap.read()
            if not ret:
                # 如果视频结束，重复最后一帧
                frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        
        # 处理帧
        inputs = self.processor(frames, return_tensors="pt")
        return inputs, start_frame

def non_max_suppression(predictions, iou_threshold=0.5):
    """
    对时序检测结果进行非极大值抑制
    Args:
        predictions: 检测结果列表，每个元素包含start_frame和end_frame
        iou_threshold: IoU阈值
    Returns:
        合并后的检测结果列表
    """
    if not predictions:
        return []
    
    # 按起始帧排序
    predictions = sorted(predictions, key=lambda x: x['start_frame'])
    merged = []
    
    while predictions:
        current = predictions.pop(0)
        
        # 计算与当前检测框的IoU
        i = 0
        while i < len(predictions):
            next_box = predictions[i]
            
            # 计算交集
            intersection_start = max(current['start_frame'], next_box['start_frame'])
            intersection_end = min(current['end_frame'], next_box['end_frame'])
            intersection = max(0, intersection_end - intersection_start)
            
            # 计算并集
            union_start = min(current['start_frame'], next_box['start_frame'])
            union_end = max(current['end_frame'], next_box['end_frame'])
            union = union_end - union_start
            
            # 计算IoU
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                # 合并重叠的检测框
                current['start_frame'] = min(current['start_frame'], next_box['start_frame'])
                current['end_frame'] = max(current['end_frame'], next_box['end_frame'])
                predictions.pop(i)
            else:
                i += 1
        
        merged.append(current)
    
    return merged

def detect_violence_scenes(video_path, model, threshold=0.5):
    dataset = LongVideoDataset(video_path)
    predictions = []
    
    for i in range(len(dataset)):
        inputs, start_frame = dataset[i]
        with torch.no_grad():
            classification, timestamps = model(inputs["pixel_values"])
            
            if classification[0][1] > threshold:  # 如果检测到打斗
                start_time = timestamps[0][0].item()
                end_time = timestamps[0][1].item()
                predictions.append({
                    'start_frame': start_frame + int(start_time * dataset.window_size),
                    'end_frame': start_frame + int(end_time * dataset.window_size)
                })
    
    # 使用NMS合并重叠的检测结果
    merged_predictions = non_max_suppression(predictions, iou_threshold=0.5)
    return merged_predictions

if __name__ == "__main__":
    train() 