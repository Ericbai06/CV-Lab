import os
import shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 配置路径
dataset_dir = Path("/Users/eric/Desktop/Python/CV/检测 玻璃瓶")
train_img_dir = dataset_dir / "train" / "images"
train_lbl_dir = dataset_dir / "train" / "labels"
val_img_dir = dataset_dir / "val" / "images"
val_lbl_dir = dataset_dir / "val" / "labels"

# 设置类别映射（从COCO到自定义类别）
# COCO数据集中"bottle"的类别ID是39
# 我们需要将其映射到我们的类别: 0(水滴形), 1(椭圆形), 2(圆形)
# 注意：这里初始映射所有瓶子为类别0，后面会通过图像分析来细分不同形状
coco_to_custom = {
    39: 0  # COCO: bottle -> 自定义: 水滴形 (初始映射，稍后细分)
}

bottle_id = None  # 全局变量

def auto_labeling():
    """使用预训练的YOLOv8x模型自动生成边界框标注"""
    print("加载YOLOv8x模型...")
    model = YOLO('yolov8x.pt')  # 使用最大的模型以获得最好的效果
    
    # 检查模型是否包含瓶子类别
    if 'bottle' not in model.names.values():
        print(f"错误：模型不包含'bottle'类别！可用类别: {model.names}")
        return
    else:
        global bottle_id
        bottle_id = [k for k, v in model.names.items() if v == 'bottle'][0]
        print(f"模型中'bottle'的类别ID为: {bottle_id}")
    
    # 处理训练集
    process_dataset(model, train_img_dir, train_lbl_dir, "训练集", bottle_id)
    
    # 处理验证集
    process_dataset(model, val_img_dir, val_lbl_dir, "验证集", bottle_id)
    
    print("\n自动标注完成！现在可以使用")
    print("python 玻璃瓶检测.py")
    print("运行训练了，或使用标注修正工具检查并改进标注质量。")

def process_dataset(model, img_dir, lbl_dir, dataset_name, bottle_id):
    """处理一个数据集（训练集或验证集）"""
    print(f"\n处理{dataset_name}图像...")
    
    if not img_dir.exists():
        print(f"错误：图像目录不存在: {img_dir}")
        return
        
    # 确保标签目录存在
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"错误：在 {img_dir} 中未找到图像文件")
        return
        
    print(f"找到 {len(img_files)} 个图像文件进行处理")
    
    # 对每个图像执行预测
    success_count = 0
    for i, img_file in enumerate(img_files):
        img_path = img_dir / img_file
        lbl_path = lbl_dir / (Path(img_file).stem + '.txt')
        
        # 跳过已标注的文件（除非强制覆盖）
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            success_count += 1
            if (i+1) % 50 == 0:
                print(f"进度: {i+1}/{len(img_files)} - 跳过已标注文件: {img_file}")
            continue
            
        try:
            # 运行预测
            results = model.predict(str(img_path), conf=0.3, verbose=False)[0]
            
            # 提取瓶子的检测结果
            bottle_detections = []
            for box in results.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                
                # 只保留瓶子检测，使用之前找到的bottle_id
                if cls == bottle_id and conf > 0.3:
                    # 获取边界框坐标并转换为YOLO格式 (x_center, y_center, width, height)
                    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                    img_w, img_h = results.orig_shape[1], results.orig_shape[0]
                    
                    # 转换为YOLO格式
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # 通过图像分析判断瓶子类型
                    # 注意：这里使用简单的高宽比来分类，实际上您可能需要更复杂的逻辑
                    bottle_type = classify_bottle_shape(img_path, (x1, y1, x2, y2))
                    bottle_detections.append((bottle_type, x_center, y_center, width, height))
            
            # 写入YOLO格式标签文件
            if bottle_detections:
                with open(lbl_path, 'w') as f:
                    for det in bottle_detections:
                        cls, x_c, y_c, w, h = det
                        f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                success_count += 1
                
            # 检测瓶口 (单独的模型或分析)
            detect_bottle_neck(img_path, lbl_path, bottle_detections)
                
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
        
        # 显示进度
        if (i+1) % 20 == 0:
            print(f"进度: {i+1}/{len(img_files)} - 成功: {success_count}")
    
    print(f"{dataset_name}处理完成。成功标注: {success_count}/{len(img_files)}")

def classify_bottle_shape(img_path, box):
    """
    分类瓶子形状：水滴形(0)、椭圆形(1)或圆形(2)
    这里使用简单的宽高比启发式方法，您可能需要更复杂的图像分析
    """
    try:
        img = cv2.imread(str(img_path))
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped = img[y1:y2, x1:x2]
        
        # 获取瓶子区域的高宽比
        h, w = cropped.shape[:2]
        aspect_ratio = w / h
        
        # 非常简单的分类逻辑 - 您可能需要更复杂的算法
        if aspect_ratio < 0.4:  # 非常窄的瓶子
            return 0  # 水滴形
        elif aspect_ratio < 0.6:  # 中等宽度
            return 1  # 椭圆形
        else:  # 宽瓶子
            return 2  # 圆形
            
    except Exception as e:
        print(f"分类瓶子形状时出错: {e}")
        return 0  # 默认为水滴形

def detect_bottle_neck(img_path, lbl_path, bottle_detections):
    """
    检测瓶口并添加到标签文件
    这是一个简化版本，假设瓶口在瓶子顶部
    您可能需要更精确的模型或分析来检测瓶口
    """
    try:
        if not bottle_detections:
            return
            
        # 仅为最大的瓶子添加瓶口
        bottles = sorted(bottle_detections, key=lambda x: x[3] * x[4], reverse=True)
        _, x_c, y_c, w, h = bottles[0]
        
        # 假设瓶口在瓶子顶部1/5的位置，宽度是瓶子的一半
        neck_w = w * 0.5
        neck_h = h * 0.2
        neck_y = y_c - (h/2) + (neck_h/2)  # 位于瓶子顶部
        
        # 添加瓶口标签 (类别ID 3)
        with open(lbl_path, 'a') as f:
            f.write(f"3 {x_c:.6f} {neck_y:.6f} {neck_w:.6f} {neck_h:.6f}\n")
            
    except Exception as e:
        print(f"检测瓶口时出错: {e}")

def check_label_distribution():
    """检查标签分布情况"""
    def count_labels(label_dir):
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        file_count = 0
        
        for lbl_file in os.listdir(label_dir):
            if not lbl_file.endswith('.txt'):
                continue
                
            file_count += 1
            with open(os.path.join(label_dir, lbl_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        if cls in class_counts:
                            class_counts[cls] += 1
                            
        return class_counts, file_count
    
    train_counts, train_files = count_labels(train_lbl_dir)
    val_counts, val_files = count_labels(val_lbl_dir)
    
    print("\n--- 标签分布 ---")
    print("训练集:")
    print(f"文件总数: {train_files}")
    print(f"水滴形截面的玻璃瓶 (0): {train_counts[0]}")
    print(f"椭圆形截面的玻璃瓶 (1): {train_counts[1]}")
    print(f"圆形截面的玻璃瓶 (2): {train_counts[2]}")
    print(f"瓶口 (3): {train_counts[3]}")
    
    print("\n验证集:")
    print(f"文件总数: {val_files}")
    print(f"水滴形截面的玻璃瓶 (0): {val_counts[0]}")
    print(f"椭圆形截面的玻璃瓶 (1): {val_counts[1]}")
    print(f"圆形截面的玻璃瓶 (2): {val_counts[2]}")
    print(f"瓶口 (3): {val_counts[3]}")
    
    # 绘图显示
    classes = ['水滴形', '椭圆形', '圆形', '瓶口']
    train_values = [train_counts[i] for i in range(4)]
    val_values = [val_counts[i] for i in range(4)]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_values, width, label='训练集')
    plt.bar(x + width/2, val_values, width, label='验证集')
    
    plt.xlabel('类别')
    plt.ylabel('标注数量')
    plt.title('数据集标注分布')
    plt.xticks(x, classes)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(dataset_dir / "label_distribution.png")
    plt.show()

if __name__ == "__main__":
    # 安装所需依赖
    try:
        import ultralytics
    except ImportError:
        print("安装必要的依赖...")
        os.system("pip install ultralytics opencv-python matplotlib")
        print("依赖安装完成。")
    
    print("=== 玻璃瓶自动标注工具 ===")
    print("1. 开始自动标注")
    print("2. 检查标签分布")
    print("q. 退出")
    
    choice = input("请选择操作: ")
    
    if choice == '1':
        auto_labeling()
        # 标注完成后检查分布
        check_label_distribution()
    elif choice == '2':
        check_label_distribution()
    elif choice.lower() == 'q':
        print("退出程序")
    else:
        print("无效选择")
