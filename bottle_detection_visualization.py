from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import sys

# 导入bottle_detector.py中的BottleDetectorApp类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bottle_detector import BottleDetectorApp

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

class BottleDetector:
    def __init__(self, model_path):
        """创建一个临时的TK根窗口来初始化检测器"""
        import tkinter as tk
        root = tk.Tk()
        self.app = BottleDetectorApp(root, model_path)
        root.withdraw()  # 隐藏窗口
    
    def predict(self, image_path):
        """使用bottle_detector的方法进行预测"""
        # 设置当前图像
        self.app.current_image_path = image_path
        self.app.current_image = cv2.imread(image_path)
        
        # 运行原始检测方法
        try:
            # 这里使用原始的model.predict方法，保持与bottle_detector一致
            results = self.app.model.predict(source=image_path, conf=0.5)
            return results
        except Exception as e:
            print(f"检测出错: {str(e)}")
            return None

def visualize_detections(model_path, image_dir, output_dir, num_samples=5, conf=0.5, iou=0.45):
    """
    可视化检测结果
    
    参数:
        model_path: 模型路径
        image_dir: 图片目录
        output_dir: 输出目录
        num_samples: 每类随机抽取的样本数
        conf: 置信度阈值
        iou: NMS的IoU阈值
    """
    # 创建检测器
    print(f"正在加载模型: {model_path}")
    detector = BottleDetector(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子目录（每个类别一个子目录）
    subdirs = [d for d in os.listdir(image_dir) 
              if os.path.isdir(os.path.join(image_dir, d)) and not d.startswith('.') and d != "mini"]
    
    print(f"找到以下类别目录: {subdirs}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(image_dir, subdir)
        
        # 查找该目录下的所有图片
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(Path(subdir_path).glob(f'*{ext}')))
            image_files.extend(list(Path(subdir_path).glob(f'*{ext.upper()}')))
        
        print(f"在 {subdir} 中找到 {len(image_files)} 张图片")
        
        # 如果没有图片，跳过
        if not image_files:
            continue
            
        # 随机选择样本
        if len(image_files) > num_samples:
            selected_files = random.sample(image_files, num_samples)
        else:
            selected_files = image_files
            
        # 为每个类别创建一个子目录
        class_output_dir = os.path.join(output_dir, subdir)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # 处理选中的图片
        for img_file in tqdm(selected_files, desc=f"处理 {subdir} 类别"):
            img_path = str(img_file)
            img_name = os.path.basename(img_path)
            
            # 读取图片
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 使用bottle_detector的predict方法
            results = detector.predict(img_path)
            
            # 创建matplotlib图像
            plt.figure(figsize=(12, 8))
            
            # 显示原始图像
            plt.imshow(img_rgb)
            
            # 如果有检测结果
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取类别ID和置信度
                        cls_id = int(box.cls[0].item())
                        conf_val = box.conf[0].item()
                        
                        # 获取类别名称
                        if hasattr(result, 'names') and cls_id in result.names:
                            cls_name = result.names[cls_id]
                        else:
                            cls_name = f"类别{cls_id}"
                        
                        # 绘制边界框
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor=plt.cm.tab10(cls_id % 10), 
                                           linewidth=2)
                        plt.gca().add_patch(rect)
                        
                        # 添加类别标签和置信度
                        plt.text(x1, y1-5, f"{cls_name} {conf_val:.2f}", 
                                color='white', fontsize=10, 
                                bbox=dict(facecolor=plt.cm.tab10(cls_id % 10), alpha=0.7))
            
            # 设置标题
            plt.title(f"检测结果: {subdir}/{img_name}")
            plt.axis('off')
            
            # 保存图像
            save_path = os.path.join(class_output_dir, f"viz_{img_name}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
    print(f"\n所有类别的可视化结果已保存到: {output_dir}")

def create_combined_visualization(results_dir, output_path):
    """
    创建汇总可视化图表
    
    参数:
        results_dir: 结果目录
        output_path: 输出文件路径
    """
    # 查找所有可视化图像
    viz_images = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith('viz_') and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                viz_images.append(os.path.join(root, file))
    
    # 如果没有找到图像，返回
    if not viz_images:
        print("未找到可视化结果图像")
        return
    
    # 随机选择一部分图像（如果太多）
    max_images = 12
    if len(viz_images) > max_images:
        viz_images = random.sample(viz_images, max_images)
    
    # 计算网格布局
    n = len(viz_images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    # 创建网格图表
    plt.figure(figsize=(15, 4 * rows))
    
    for i, img_path in enumerate(viz_images):
        # 读取图像
        img = plt.imread(img_path)
        
        # 创建子图
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(os.path.dirname(img_path)) + '/' + os.path.basename(img_path))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"汇总可视化图表已保存到: {output_path}")

if __name__ == "__main__":
    # 配置参数
    model_path = "/Users/eric/Desktop/Python/CV/run_93/best.pt"
    image_dir = "/Users/eric/Desktop/Python/CV/检测 玻璃瓶"
    output_dir = "/Users/eric/Desktop/Python/CV/visualization_results"
    
    # 询问用户是否要调整参数
    print("玻璃瓶检测可视化程序")
    print("-" * 30)
    
    use_default = input("是否使用默认参数? (y/n，默认y): ").strip().lower() != 'n'
    
    if not use_default:
        model_path = input(f"请输入模型路径 (默认: {model_path}): ").strip() or model_path
        image_dir = input(f"请输入图片目录 (默认: {image_dir}): ").strip() or image_dir
        output_dir = input(f"请输入输出目录 (默认: {output_dir}): ").strip() or output_dir
        
        num_samples = 5
        num_samples_input = input(f"请输入每类随机抽取的样本数 (默认: {num_samples}): ").strip()
        if num_samples_input:
            try:
                num_samples = int(num_samples_input)
            except ValueError:
                print(f"无效的样本数值，使用默认值: {num_samples}")
        
        conf_threshold = 0.5
        conf_input = input(f"请输入置信度阈值 (0-1，默认: {conf_threshold}): ").strip()
        if conf_input:
            try:
                conf_threshold = float(conf_input)
                if conf_threshold < 0 or conf_threshold > 1:
                    print(f"置信度必须在0-1之间，使用默认值: {conf_threshold}")
                    conf_threshold = 0.5
            except ValueError:
                print(f"无效的置信度值，使用默认值: {conf_threshold}")
                
        iou_threshold = 0.45
        iou_input = input(f"请输入IoU阈值 (0-1，默认: {iou_threshold}): ").strip()
        if iou_input:
            try:
                iou_threshold = float(iou_input)
                if iou_threshold < 0 or iou_threshold > 1:
                    print(f"IoU阈值必须在0-1之间，使用默认值: {iou_threshold}")
                    iou_threshold = 0.45
            except ValueError:
                print(f"无效的IoU阈值，使用默认值: {iou_threshold}")
    else:
        num_samples = 5
        conf_threshold = 0.5
        iou_threshold = 0.45
    
    # 运行可视化
    print("\n开始可视化检测结果...")
    visualize_detections(model_path, image_dir, output_dir, num_samples, conf_threshold, iou_threshold)
    
    # 创建汇总可视化
    summary_path = os.path.join(output_dir, 'detection_summary.png')
    create_combined_visualization(output_dir, summary_path)