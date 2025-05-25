#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频打斗场景检测GUI程序
"""

import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QSlider,
                            QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import av
import logging
import time
import json
from datetime import datetime

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    """视频处理线程"""
    frame_signal = pyqtSignal(np.ndarray, float, float)  # 帧图像, 置信度, 时间戳
    detection_signal = pyqtSignal(float, float, float)  # 开始时间, 结束时间, 置信度
    
    def __init__(self, video_path, model, processor, window_size=16, stride=8):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.processor = processor
        self.window_size = window_size
        self.stride = stride
        self.running = True
        self.frame_buffer = []
        self.frame_count = 0
        self.confidence_threshold = 0.6
        self.detection_segments = []
        self.current_segment = None
        
    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = threshold
        
    def run(self):
        try:
            # 打开视频文件
            container = av.open(self.video_path, options={
                'threads': '0',
                'skip_frame': 'nokey',
                'flags': 'low_delay',
                'strict': 'experimental'
            })
            
            # 获取视频信息
            stream = container.streams.video[0]
            logger.info(f"average_rate: {stream.average_rate} ({type(stream.average_rate)})")
            logger.info(f"duration: {stream.duration} ({type(stream.duration)})")
            logger.info(f"time_base: {stream.time_base} ({type(stream.time_base)})")
            logger.info(f"frames: {stream.frames} ({type(stream.frames)})")
            logger.info(f"start_time: {stream.start_time} ({type(stream.start_time)})")
            fps = float(stream.average_rate)
            
            for frame in container.decode(video=0):
                if not self.running:
                    break
                    
                # 处理帧
                frame_array = frame.to_ndarray(format="rgb24")
                self.frame_count += 1
                current_time = self.frame_count / fps
                
                # 预处理帧
                processed_frame = self.preprocess_frame(frame_array)
                self.frame_buffer.append(processed_frame)
                
                # 当缓冲区达到窗口大小时进行预测
                if len(self.frame_buffer) >= self.window_size:
                    # 准备批处理数据
                    batch_frames = torch.stack(self.frame_buffer[:self.window_size])
                    batch_frames = batch_frames.unsqueeze(0).to(self.model.device)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = self.model(pixel_values=batch_frames)
                        probs = torch.softmax(outputs.logits, dim=1)
                        confidence = probs[0][1].item()
                    
                    # 处理检测结果
                    if confidence > self.confidence_threshold:
                        if self.current_segment is None:
                            # 开始新的检测段
                            self.current_segment = {
                                'start_time': current_time,
                                'end_time': current_time,
                                'confidence': confidence
                            }
                        else:
                            # 更新当前检测段
                            self.current_segment['end_time'] = current_time
                            self.current_segment['confidence'] = max(
                                self.current_segment['confidence'], confidence)
                    else:
                        if self.current_segment is not None:
                            # 结束当前检测段
                            self.detection_segments.append(self.current_segment)
                            self.detection_signal.emit(
                                self.current_segment['start_time'],
                                self.current_segment['end_time'],
                                self.current_segment['confidence']
                            )
                            self.current_segment = None
                    
                    # 发送帧和预测结果
                    self.frame_signal.emit(frame_array, confidence, current_time)
                    
                    # 更新缓冲区
                    self.frame_buffer = self.frame_buffer[self.stride:]
                
                # 控制处理速度
                time.sleep(1/fps)
            
            # 处理最后一个检测段
            if self.current_segment is not None:
                self.detection_segments.append(self.current_segment)
                self.detection_signal.emit(
                    self.current_segment['start_time'],
                    self.current_segment['end_time'],
                    self.current_segment['confidence']
                )
            
            container.close()
            
        except Exception as e:
            logger.error(f"视频处理错误: {str(e)}")
            
    def stop(self):
        self.running = False
        
    def preprocess_frame(self, frame):
        """预处理单个视频帧"""
        # 调整大小
        frame = cv2.resize(frame, (224, 224))
        # 转换为张量
        frame = torch.from_numpy(frame).float()
        # 调整维度顺序
        frame = frame.permute(2, 0, 1)
        # 归一化
        frame = frame / 255.0
        return frame

class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_model()
        self.detection_segments = []
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('视频打斗场景检测')
        self.setGeometry(100, 100, 1280, 720)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # 控制区域
        control_layout = QHBoxLayout()
        
        # 打开文件按钮
        self.open_button = QPushButton('打开视频')
        self.open_button.clicked.connect(self.open_video)
        control_layout.addWidget(self.open_button)
        
        # 播放/暂停按钮
        self.play_button = QPushButton('播放')
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        control_layout.addWidget(self.play_button)
        
        # 保存结果按钮
        self.save_button = QPushButton('保存结果')
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        control_layout.addWidget(self.save_button)
        
        # 置信度阈值控制
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel('置信度阈值:'))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setValue(0.6)
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_spinbox)
        control_layout.addLayout(threshold_layout)
        
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        control_layout.addWidget(self.slider)
        
        # 状态标签
        self.status_label = QLabel('就绪')
        control_layout.addWidget(self.status_label)
        
        layout.addLayout(control_layout)
        
        # 检测结果显示区域
        self.result_label = QLabel('检测结果:')
        layout.addWidget(self.result_label)
        
        main_widget.setLayout(layout)
        
        # 初始化变量
        self.video_thread = None
        self.is_playing = False
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info("加载模型...")
            self.processor = VideoMAEImageProcessor.from_pretrained("./best_model/model")
            self.model = VideoMAEForVideoClassification.from_pretrained("./best_model/model")
            self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.eval()
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            self.status_label.setText('模型加载失败')
            
    def update_threshold(self, value):
        """更新置信度阈值"""
        if self.video_thread:
            self.video_thread.set_confidence_threshold(value)
            
    def open_video(self):
        """打开视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "", 
                                                 "视频文件 (*.mp4 *.mkv *.avi)")
        if file_name:
            self.video_path = file_name
            self.play_button.setEnabled(True)
            self.slider.setEnabled(True)
            self.save_button.setEnabled(True)
            self.status_label.setText('视频已加载')
            self.detection_segments = []
            self.result_label.setText('检测结果:')
            
    def toggle_play(self):
        """切换播放/暂停状态"""
        if not self.is_playing:
            self.start_video()
        else:
            self.stop_video()
            
    def start_video(self):
        """开始播放视频"""
        if hasattr(self, 'video_path'):
            self.video_thread = VideoThread(self.video_path, self.model, self.processor)
            self.video_thread.frame_signal.connect(self.update_frame)
            self.video_thread.detection_signal.connect(self.update_detection_results)
            self.video_thread.set_confidence_threshold(self.threshold_spinbox.value())
            self.video_thread.start()
            self.is_playing = True
            self.play_button.setText('暂停')
            self.status_label.setText('正在播放')
            
    def stop_video(self):
        """停止播放视频"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.is_playing = False
            self.play_button.setText('播放')
            self.status_label.setText('已暂停')
            
    def update_frame(self, frame, confidence, timestamp):
        """更新视频帧和检测结果"""
        # 在帧上添加检测结果
        h, w = frame.shape[:2]
        text = f"置信度: {confidence:.2%}"
        color = (0, 255, 0) if confidence < self.threshold_spinbox.value() else (0, 0, 255)
        
        # 添加文本背景
        cv2.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 添加时间戳
        time_str = f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{int(timestamp%60):02d}"
        cv2.putText(frame, time_str, (w-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 转换为QImage并显示
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整图像大小以适应窗口
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_detection_results(self, start_time, end_time, confidence):
        """更新检测结果"""
        segment = {
            'start_time': start_time,
            'end_time': end_time,
            'confidence': confidence
        }
        self.detection_segments.append(segment)
        
        # 更新显示
        start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
        end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
        result_text = f"检测到打斗场景: {start_str} - {end_str} (置信度: {confidence:.2%})"
        self.result_label.setText(result_text)
        
    def save_results(self):
        """保存检测结果"""
        if not self.detection_segments:
            self.status_label.setText('没有检测结果可保存')
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", 
                                                 "文本文件 (*.txt);;JSON文件 (*.json)")
        if file_name:
            try:
                if file_name.endswith('.json'):
                    # 保存为JSON格式
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump({
                            'video_path': self.video_path,
                            'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'threshold': self.threshold_spinbox.value(),
                            'segments': self.detection_segments
                        }, f, indent=4, ensure_ascii=False)
                else:
                    # 保存为文本格式
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write(f"视频文件: {self.video_path}\n")
                        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"置信度阈值: {self.threshold_spinbox.value()}\n\n")
                        f.write("检测到的打斗场景:\n")
                        for i, segment in enumerate(self.detection_segments, 1):
                            start_str = f"{int(segment['start_time']//3600):02d}:{int((segment['start_time']%3600)//60):02d}:{int(segment['start_time']%60):02d}"
                            end_str = f"{int(segment['end_time']//3600):02d}:{int((segment['end_time']%3600)//60):02d}:{int(segment['end_time']%60):02d}"
                            f.write(f"{i}. {start_str} - {end_str} (置信度: {segment['confidence']:.2%})\n")
                
                self.status_label.setText('检测结果已保存')
            except Exception as e:
                logger.error(f"保存结果失败: {str(e)}")
                self.status_label.setText('保存失败')
        
    def closeEvent(self, event):
        """关闭窗口事件"""
        self.stop_video()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 