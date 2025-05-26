import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from transformers import VideoMAEForVideoClassification
import time
from datetime import timedelta

# 设置环境变量来配置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Define class mapping
class_mapping = {
    "Abuse": 0, "Arrest": 1, "Arson": 2, "Assault": 3, "Burglary": 4,
    "Explosion": 5, "Fighting": 6, "Normal Videos": 7, "Road Accidents": 8,
    "Robbery": 9, "Shooting": 10, "Shoplifting": 11, "Stealing": 12, "Vandalism": 13
}
reverse_mapping = {v: k for k, v in class_mapping.items()}

# Load VideoMAE model
model_name = "OPear/videomae-large-finetuned-UCF-Crime"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VideoMAEForVideoClassification.from_pretrained(
    model_name,
    label2id=class_mapping,
    id2label=reverse_mapping,
    ignore_mismatched_sizes=True,
).to(device)
model.eval()

def process_frame(frame, size=(224, 224)):
    """处理单帧图像"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return frame

def detect_scene_change(frame1, frame2, threshold=30.0):
    """检测场景变化"""
    if frame1 is None or frame2 is None:
        return True
    
    diff = cv2.absdiff(frame1, frame2)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def format_time(seconds):
    """将秒数转换为时:分:秒格式"""
    return str(timedelta(seconds=int(seconds)))

def predict_video_stream(video_path, confidence_threshold=0.5, scene_change_threshold=30.0):
    """实时视频流预测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # 获取视频FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_buffer = []
    prev_frame = None
    current_scene = None
    current_confidence = 0.0
    last_prediction_time = time.time()
    prediction_interval = 1.0  # 每秒预测一次
    
    # 时间记录相关变量
    current_segment_start = None
    high_confidence_segments = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前时间戳
        current_time = frame_count / fps
        frame_count += 1

        # 检测场景变化
        if detect_scene_change(prev_frame, frame, scene_change_threshold):
            # 如果当前有正在记录的高置信度片段，记录结束时间
            if current_segment_start is not None:
                high_confidence_segments.append({
                    'scene': current_scene,
                    'confidence': current_confidence,
                    'start_time': current_segment_start,
                    'end_time': current_time
                })
                print(f"\n场景变化 - 片段结束:")
                print(f"场景: {current_scene}")
                print(f"置信度: {current_confidence:.2f}")
                print(f"时间段: {format_time(current_segment_start)} - {format_time(current_time)}")
                current_segment_start = None
            
            frame_buffer = []
            current_scene = None
            current_confidence = 0.0

        # 处理当前帧
        processed_frame = process_frame(frame)
        frame_buffer.append(processed_frame)

        # 保持固定帧数
        if len(frame_buffer) > 16:
            frame_buffer.pop(0)

        # 定期进行预测
        current_time = time.time()
        if len(frame_buffer) == 16 and (current_time - last_prediction_time) >= prediction_interval:
            # 准备输入张量
            video_tensor = torch.stack(frame_buffer, dim=0)
            video_tensor = video_tensor.unsqueeze(0).to(device)

            # 进行预测
            with torch.no_grad():
                outputs = model(video_tensor)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_label = torch.max(probs, dim=-1)
                confidence = confidence.item()
                predicted_label = predicted_label.item()

            # 更新当前场景
            if confidence > confidence_threshold:
                if current_scene != reverse_mapping[predicted_label]:
                    # 如果场景改变，记录新的开始时间
                    current_segment_start = frame_count / fps
                current_scene = reverse_mapping[predicted_label]
                current_confidence = confidence
            else:
                # 如果置信度低于阈值，结束当前片段
                if current_segment_start is not None:
                    high_confidence_segments.append({
                        'scene': current_scene,
                        'confidence': current_confidence,
                        'start_time': current_segment_start,
                        'end_time': frame_count / fps
                    })
                    print(f"\n置信度降低 - 片段结束:")
                    print(f"场景: {current_scene}")
                    print(f"置信度: {current_confidence:.2f}")
                    print(f"时间段: {format_time(current_segment_start)} - {format_time(frame_count / fps)}")
                    current_segment_start = None
                current_scene = None
                current_confidence = 0.0

            last_prediction_time = current_time

        # 显示结果
        if current_scene:
            text = f"Scene: {current_scene} (Confidence: {current_confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 显示当前时间
            time_text = f"Time: {format_time(frame_count / fps)}"
            cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示视频帧
        cv2.imshow('Video Analysis', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame.copy()

    # 视频结束时，如果还有未结束的片段，记录它
    if current_segment_start is not None:
        high_confidence_segments.append({
            'scene': current_scene,
            'confidence': current_confidence,
            'start_time': current_segment_start,
            'end_time': frame_count / fps
        })
        print(f"\n视频结束 - 最后片段:")
        print(f"场景: {current_scene}")
        print(f"置信度: {current_confidence:.2f}")
        print(f"时间段: {format_time(current_segment_start)} - {format_time(frame_count / fps)}")

    # 打印所有高置信度片段摘要
    print("\n=== 高置信度片段摘要 ===")
    for i, segment in enumerate(high_confidence_segments, 1):
        print(f"\n片段 {i}:")
        print(f"场景: {segment['scene']}")
        print(f"置信度: {segment['confidence']:.2f}")
        print(f"时间段: {format_time(segment['start_time'])} - {format_time(segment['end_time'])}")
        print(f"持续时间: {format_time(segment['end_time'] - segment['start_time'])}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "your_video_path.mp4"  # 替换为你的视频路径
    predict_video_stream(video_path)
