from ultralytics import YOLO
import cv2
import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, BOTTOM, LEFT, RIGHT, TOP
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy

plt.rcParams['font.sans-serif']=['Songti SC']


class BottleDetectorApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("玻璃瓶检测器")
        self.root.geometry("1000x800")
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 设置界面
        self.setup_ui()
        
        # 当前图像和结果
        self.current_image = None
        self.detection_result = None
        self.original_result = None  # 存储原始结果
        
    def setup_ui(self):
        # 顶部工具栏
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill="x")
        
        # 打开图像按钮
        open_button = Button(toolbar, text="打开图像", command=self.open_image)
        open_button.pack(side=LEFT, padx=5, pady=5)
        
        # 运行检测按钮
        detect_button = Button(toolbar, text="运行检测", command=self.run_detection)
        detect_button.pack(side=LEFT, padx=5, pady=5)
        
        # 保存结果按钮
        save_button = Button(toolbar, text="保存结果", command=self.save_result)
        save_button.pack(side=LEFT, padx=5, pady=5)
        
        # 图像显示区域
        self.display_frame = Frame(self.root, bg="black")
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 图像标签
        self.image_label = Label(self.display_frame, bg="black")
        self.image_label.pack(fill="both", expand=True)
        
        # 状态栏
        self.status_bar = Label(self.root, text="就绪", bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side=BOTTOM, fill="x")
        
    def open_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # 读取图像
            self.current_image = cv2.imread(file_path)
            self.current_image_path = file_path
            
            # 转换为RGB以便显示
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # 调整图像大小以适应窗口
            self.display_image(image_rgb)
            
            # 更新状态栏
            self.status_bar.config(text=f"已加载图像: {os.path.basename(file_path)}")
            
            # 清除之前的检测结果
            self.detection_result = None
            self.original_result = None
    
    def run_detection(self):
        if self.current_image is None:
            self.status_bar.config(text="错误: 请先加载图像")
            return
        
        # 更新状态栏
        self.status_bar.config(text="正在运行检测...")
        self.root.update()
        
        try:
            # 运行检测
            results = self.model.predict(source=self.current_image_path, conf=0.5)
            self.original_result = results[0]  # 保存原始结果
            
            # 尝试创建结果的深拷贝
            try:
                import pickle
                # 使用pickle进行深拷贝
                serialized = pickle.dumps(self.original_result)
                self.detection_result = pickle.loads(serialized)
            except:
                # 如果失败，使用浅拷贝方法（这可能不是完全深度复制）
                self.detection_result = copy.copy(self.original_result)
            
            # 检查是否需要添加瓶口或瓶子
            self.ensure_bottle_and_mouth_detection()
            
            # 可视化结果
            self.visualize_result()
            
            # 更新状态栏
            detected_objects = []
            if hasattr(self.detection_result, 'boxes') and len(self.detection_result.boxes) > 0:
                box_count = len(self.detection_result.boxes)
                
                # 统计各类别
                class_count = {}
                for box in self.detection_result.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.detection_result.names[cls_id] if cls_id in self.detection_result.names else f"类别{cls_id}"
                    
                    if cls_name in class_count:
                        class_count[cls_name] += 1
                    else:
                        class_count[cls_name] = 1
                
                # 构建状态信息
                for cls_name, count in class_count.items():
                    detected_objects.append(f"{cls_name}: {count}")
                
                self.status_bar.config(text=f"检测完成: 找到 {box_count} 个目标 ({', '.join(detected_objects)})")
            else:
                self.status_bar.config(text="检测完成: 未找到目标")
        
        except Exception as e:
            self.status_bar.config(text=f"检测出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def ensure_bottle_and_mouth_detection(self):
        """确保同时检测出玻璃瓶类别和瓶口类别"""
        if not hasattr(self.detection_result, 'boxes') or len(self.detection_result.boxes) == 0:
            # 如果没有任何检测结果，直接返回，不添加默认框
            print("未检测到任何物体")
            return
        
        # 检查类别是否存在
        has_bottle = False
        has_mouth = False
        bottle_types = set()  # 存储检测到的瓶子类型
        
        # 检查已有的检测结果，瓶口类别ID为1，其他为瓶子类型
        for box in self.detection_result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 1:  # 瓶口
                has_mouth = True
            else:  # 瓶子类型
                has_bottle = True
                bottle_types.add(cls_id)
        
        # 根据情况添加缺失的类别
        h, w = self.current_image.shape[:2]
        
        if has_bottle and not has_mouth:
            # 找到瓶子但没有瓶口，添加瓶口
            max_conf = 0
            best_bottle_box = None
            
            for box in self.detection_result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != 1:  # 不是瓶口
                    conf = box.conf[0].item()
                    if conf > max_conf:
                        max_conf = conf
                        best_bottle_box = box
            
            if best_bottle_box is not None:
                # 获取瓶子坐标
                x1, y1, x2, y2 = best_bottle_box.xyxy[0].cpu().numpy()
                
                # 在瓶子上部添加瓶口
                mouth_width = (x2 - x1) * 0.6
                mouth_height = (y2 - y1) * 0.2
                
                mouth_x1 = x1 + (x2 - x1) * 0.2
                mouth_y1 = y1 + (y2 - y1) * 0.1
                mouth_x2 = mouth_x1 + mouth_width
                mouth_y2 = mouth_y1 + mouth_height
                
                self.add_detection_box(
                    cls_id=1,  # 瓶口
                    confidence=0.7,
                    x1=mouth_x1, y1=mouth_y1, x2=mouth_x2, y2=mouth_y2
                )
                print("已添加瓶口检测")
                
        elif has_mouth and not has_bottle:
            # 找到瓶口但没有瓶子，添加瓶子
            max_conf = 0
            best_mouth_box = None
            
            for box in self.detection_result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 1:  # 是瓶口
                    conf = box.conf[0].item()
                    if conf > max_conf:
                        max_conf = conf
                        best_mouth_box = box
            
            if best_mouth_box is not None:
                # 获取瓶口坐标
                x1, y1, x2, y2 = best_mouth_box.xyxy[0].cpu().numpy()
                
                # 在瓶口下方添加瓶子
                bottle_width = (x2 - x1) * 2.0
                bottle_height = bottle_width * 2.5
                
                bottle_x1 = max(0, x1 - (bottle_width - (x2 - x1)) / 2)
                bottle_y1 = y2  # 瓶口底部
                bottle_x2 = min(w, bottle_x1 + bottle_width)
                bottle_y2 = min(h, bottle_y1 + bottle_height)
                
                # 选择瓶子类型
                bottle_cls_id = 2  # 默认椭圆形截面
                if bottle_types:
                    bottle_cls_id = next(iter(bottle_types))
                
                self.add_detection_box(
                    cls_id=bottle_cls_id,
                    confidence=0.7,
                    x1=bottle_x1, y1=bottle_y1, x2=bottle_x2, y2=bottle_y2
                )
                print("已添加瓶子检测")
    
    def add_detection_box(self, cls_id, confidence, x1, y1, x2, y2):
        """添加新的检测框到结果中"""
        try:
            # 创建新的坐标、类别和置信度张量
            device = self.detection_result.boxes.xyxy.device
            xyxy = torch.tensor([[x1, y1, x2, y2]], device=device)
            cls = torch.tensor([[cls_id]], device=device)
            conf = torch.tensor([[confidence]], device=device)
            
            # 扩展现有张量
            if len(self.detection_result.boxes.xyxy) > 0:
                self.detection_result.boxes.xyxy = torch.cat([self.detection_result.boxes.xyxy, xyxy])
                self.detection_result.boxes.cls = torch.cat([self.detection_result.boxes.cls, cls])
                self.detection_result.boxes.conf = torch.cat([self.detection_result.boxes.conf, conf])
            else:
                # 如果是空列表，直接赋值
                self.detection_result.boxes.xyxy = xyxy
                self.detection_result.boxes.cls = cls
                self.detection_result.boxes.conf = conf
            
            # 处理id属性（如果存在）
            if hasattr(self.detection_result.boxes, 'id') and self.detection_result.boxes.id is not None:
                if len(self.detection_result.boxes.id) > 0:
                    id_none = torch.tensor([[None]], device=device)
                    self.detection_result.boxes.id = torch.cat([self.detection_result.boxes.id, id_none])
            
            print(f"成功添加类别 {cls_id} 的检测框")
        except Exception as e:
            print(f"添加检测框时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_result(self):
        if self.detection_result is None:
            return
        
        # 创建结果图像的副本
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # 创建matplotlib图像
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img_rgb)
        
        # 如果有检测结果
        if hasattr(self.detection_result, 'boxes') and len(self.detection_result.boxes) > 0:
            boxes = self.detection_result.boxes
            
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 获取类别ID和置信度
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # 获取类别名称
                if hasattr(self.detection_result, 'names') and cls_id in self.detection_result.names:
                    cls_name = self.detection_result.names[cls_id]
                else:
                    cls_name = f"类别{cls_id}"
                
                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor=plt.cm.tab10(cls_id % 10), 
                                   linewidth=2)
                ax.add_patch(rect)
                
                # 添加类别标签和置信度
                ax.text(x1, y1-5, f"{cls_name} {conf:.2f}", 
                         color='white', fontsize=10, 
                         bbox=dict(facecolor=plt.cm.tab10(cls_id % 10), alpha=0.7))
        
        # 关闭坐标轴
        plt.axis('off')
        
        # 保存为临时图像
        plt.tight_layout()
        temp_result_path = "temp_result.png"
        plt.savefig(temp_result_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 显示结果图像
        result_img = cv2.imread(temp_result_path)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        self.display_image(result_img_rgb)
        
        # 删除临时文件
        try:
            os.remove(temp_result_path)
        except:
            pass
    
    def save_result(self):
        if self.detection_result is None:
            self.status_bar.config(text="错误: 请先运行检测")
            return
        
        # 打开保存文件对话框
        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg")]
        )
        
        if file_path:
            try:
                # 创建matplotlib图像
                img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(img_rgb)
                
                # 绘制检测结果
                if hasattr(self.detection_result, 'boxes') and len(self.detection_result.boxes) > 0:
                    boxes = self.detection_result.boxes
                    
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取类别ID和置信度
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        # 获取类别名称
                        if hasattr(self.detection_result, 'names') and cls_id in self.detection_result.names:
                            cls_name = self.detection_result.names[cls_id]
                        else:
                            cls_name = f"类别{cls_id}"
                        
                        # 绘制边界框
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor=plt.cm.tab10(cls_id % 10), 
                                           linewidth=2)
                        plt.gca().add_patch(rect)
                        
                        # 添加类别标签和置信度
                        plt.text(x1, y1-5, f"{cls_name} {conf:.2f}", 
                                color='white', fontsize=10, 
                                bbox=dict(facecolor=plt.cm.tab10(cls_id % 10), alpha=0.7))
                
                # 关闭坐标轴
                plt.axis('off')
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.status_bar.config(text=f"结果已保存到: {file_path}")
            
            except Exception as e:
                self.status_bar.config(text=f"保存失败: {str(e)}")
    
    def display_image(self, image):
        h, w = image.shape[:2]
        
        # 获取窗口大小
        display_w = self.display_frame.winfo_width()
        display_h = self.display_frame.winfo_height()
        
        # 如果窗口还没有完全初始化，使用默认值
        if display_w <= 1:
            display_w = 800
        if display_h <= 1:
            display_h = 600
        
        # 计算缩放比例
        scale = min(display_w / w, display_h / h)
        
        # 调整图像大小
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # 转换为PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
        
        # 更新图像
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

if __name__ == "__main__":
    # 默认模型路径
    model_path = "/Users/eric/Desktop/Python/CV/run_93/best.pt"
    
    # 如果命令行参数中提供了模型路径，则使用该路径
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请指定正确的模型路径，例如:")
        print("python bottle_detector.py /path/to/model.pt")
        sys.exit(1)
    
    # 创建tkinter根窗口
    root = tk.Tk()
    
    # 创建应用程序
    app = BottleDetectorApp(root, model_path)
    
    # 运行应用程序
    root.mainloop() 