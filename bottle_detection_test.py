from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import sys
import pandas as pd
from collections import Counter, defaultdict

# 导入bottle_detector.py中的BottleDetectorApp类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bottle_detector import BottleDetectorApp
# 导入bottle_detection_utils.py中的检测函数
from bottle_detection_utils import detect_bottle_with_neck

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BottleDetector:
    def __init__(self, model_path):
        """创建一个临时的TK根窗口来初始化检测器"""
        import tkinter as tk
        root = tk.Tk()
        self.app = BottleDetectorApp(root, model_path)
        root.withdraw()  # 隐藏窗口
        # 直接加载模型供bottle_detection_utils.py使用
        self.model = self.app.model
    
    def predict(self, image_path):
        """使用bottle_detector的方法进行预测"""
        # 设置当前图像
        self.app.current_image_path = image_path
        self.app.current_image = cv2.imread(image_path)
        
        # 使用bottle_detection_utils.py中的函数进行检测
        # 这个函数会处理多瓶检测的情况，只保留最高置信度的瓶子
        try:
            results = detect_bottle_with_neck(
                image_path=image_path, 
                model=self.model, 
                conf=0.5,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            return results
        except Exception as e:
            print(f"检测出错: {str(e)}")
            return None

def save_detection_visualization(image_path, result, save_path, show_conf=True):
    """
    保存带有检测框的图像
    
    参数:
        image_path: 原始图像路径
        result: YOLO模型的检测结果
        save_path: 保存路径
        show_conf: 是否显示置信度
    """
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取原始图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建matplotlib图像
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    
    # 绘制检测框
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        boxes = result.boxes
        
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 获取类别ID和置信度
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
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
            label = f"{cls_name}" if not show_conf else f"{cls_name} {conf:.2f}"
            plt.text(x1, y1-5, label, 
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=plt.cm.tab10(cls_id % 10), alpha=0.7))
    
    # 关闭坐标轴
    plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_detection_results(data_dir, image_files, detection_results, class_counts, output_dir):
    """
    分析检测结果，生成统计报告
    
    参数:
        data_dir: 数据目录
        image_files: 图片文件列表
        detection_results: 检测结果列表(每项是一个字典，包含图片路径和检测结果)
        class_counts: 各类别检测数量统计
        output_dir: 输出目录
    """
    print("开始分析检测结果...")
    
    # 1. 获取预期的类别信息（直接通过文件计数，参考字体.py的方法）
    expected_counts = {}
    bottle_types = {
        "水滴形截面的玻璃瓶": "水滴形截面的玻璃瓶",
        "圆形截面的玻璃瓶": "圆形截面的玻璃瓶",
        "椭圆形截面的玻璃瓶": "椭圆形截面的玻璃瓶"
    }
    
    # 直接计算每种瓶子类型的实际图片数量
    bottle_images_by_type = {}
    for bottle_type in bottle_types.values():
        bottle_dir = os.path.join(data_dir, bottle_type)
        if os.path.isdir(bottle_dir):
            # 获取该目录下的所有图片文件
            files_count = 0
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for file in Path(bottle_dir).glob(f'*{ext}'):
                    if not any(exclude_dir in str(file) for exclude_dir in ["mini"]):
                        files_count += 1
                for file in Path(bottle_dir).glob(f'*{ext.upper()}'):
                    if not any(exclude_dir in str(file) for exclude_dir in ["mini"]):
                        files_count += 1
            
            bottle_images_by_type[bottle_type] = files_count
            expected_counts[bottle_type] = files_count
            print(f"在'{bottle_type}'目录中找到 {files_count} 张图片")
    
    # 2. 统计检测结果
    # 创建一个详细的分析数据结构
    detection_stats = {
        "总图片数": len(image_files),
        "检测出类别数": len(class_counts),
        "类别检测统计": class_counts.copy(),
        "预期类别统计": expected_counts.copy(),
        "类别误差分析": {},
        "缺失分析": {
            "未检测到任何物体的图片": [],
            "仅检测到瓶口的图片": [],
            "仅检测到瓶身的图片": [],
            "检测到多个瓶子的图片": []
        }
    }
    
    # 计算每个瓶子类型的误差率
    for class_name, expected in expected_counts.items():
        detected = class_counts.get(class_name, 0)
        if expected > 0:
            error = detected - expected
            error_rate = error / expected * 100
            detection_stats["类别误差分析"][class_name] = {
                "预期数量": expected,
                "检测数量": detected,
                "误差": error,
                "误差率": f"{error_rate:.2f}%"
            }
    
    # 计算瓶口与瓶身数量比较
    bottle_types_total = sum(count for class_name, count in class_counts.items() 
                       if class_name in bottle_types.values())
    mouth_count = class_counts.get("瓶口", 0)
    
    if bottle_types_total > 0:
        mouth_to_bottle_ratio = mouth_count / bottle_types_total
        detection_stats["瓶口与瓶身比例"] = f"{mouth_to_bottle_ratio:.2f}"
        
        if mouth_count > bottle_types_total:
            detection_stats["瓶口过多分析"] = f"检测到的瓶口数量({mouth_count})多于瓶身总数({bottle_types_total})，可能存在重复检测或误检"
        elif mouth_count < bottle_types_total:
            detection_stats["瓶口不足分析"] = f"检测到的瓶口数量({mouth_count})少于瓶身总数({bottle_types_total})，可能存在瓶口漏检"
    
    # 3. 图片级别的详细分析
    per_image_stats = []
    
    # 按瓶子类型对图片进行归类
    images_by_type = defaultdict(list)
    for img_path in image_files:
        found_type = False
        # 获取目录路径
        dir_path = os.path.dirname(img_path)
        # 获取目录名称
        dir_name = os.path.basename(dir_path)
        
        # 检查目录名称是否匹配某个瓶子类型
        for bottle_type in bottle_types.values():
            if dir_name == bottle_type:
                images_by_type[bottle_type].append(img_path)
                found_type = True
                break
                
        if not found_type and "玻璃瓶" in dir_path:
            # 打印无法归类的图片路径
            print(f"警告: 无法确定瓶子类型: {img_path}")
    
    # 打印所有类型的图片数量
    print("各类别图片数量 (debug信息):")
    for bottle_type in bottle_types.values():
        count = len(images_by_type[bottle_type])
        print(f"  {bottle_type}: {count} 张图片")
    
    # 保存各类别的图片计数结果
    images_by_type_counts = {k: len(v) for k, v in images_by_type.items()}
    print("各类别图片数量:")
    for bottle_type, count in images_by_type_counts.items():
        print(f"  {bottle_type}: {count} 张图片")
    
    # 分析每张图片的检测结果
    for result_info in detection_results:
        img_path = result_info["image_path"]
        result = result_info["result"]
        
        # 获取图片所属的真实类别（基于文件路径）
        actual_class = result_info["actual_class"]
        
        # 统计该图片中检测到的类别
        detected_classes = Counter()
        bottle_body_count = 0
        mouth_count = 0
        
        if result and hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # 获取类别名称
                if hasattr(result, 'names') and cls_id in result.names:
                    cls_name = result.names[cls_id]
                else:
                    cls_name = f"类别{cls_id}"
                
                detected_classes[cls_name] += 1
                
                if cls_name == "瓶口":
                    mouth_count += 1
                else:
                    bottle_body_count += 1
        
        # 记录图片级别的统计
        img_stats = {
            "图片路径": os.path.relpath(img_path, data_dir),
            "实际类别": actual_class,
            "检测到的类别": list(detected_classes.keys()),
            "检测到的瓶子数量": bottle_body_count,
            "检测到的瓶口数量": mouth_count,
            "检测结果是否正确": False  # 默认设为False，下面会更新
        }
        
        # 分析检测结果是否正确
        if actual_class and actual_class in detected_classes:
            # 如果检测到了正确的瓶子类型
            img_stats["检测结果是否正确"] = True
        
        # 缺失分析
        if not detected_classes:
            detection_stats["缺失分析"]["未检测到任何物体的图片"].append(img_stats["图片路径"])
            img_stats["分析结果"] = "未检测到任何物体"
        elif mouth_count > 0 and bottle_body_count == 0:
            detection_stats["缺失分析"]["仅检测到瓶口的图片"].append(img_stats["图片路径"])
            img_stats["分析结果"] = "仅检测到瓶口，未检测到瓶身"
        elif mouth_count == 0 and bottle_body_count > 0:
            detection_stats["缺失分析"]["仅检测到瓶身的图片"].append(img_stats["图片路径"])
            img_stats["分析结果"] = "仅检测到瓶身，未检测到瓶口"
        elif bottle_body_count > 1:
            detection_stats["缺失分析"]["检测到多个瓶子的图片"].append(img_stats["图片路径"])
            img_stats["分析结果"] = f"检测到{bottle_body_count}个瓶身"
        else:
            img_stats["分析结果"] = "检测正常"
        
        per_image_stats.append(img_stats)
    
    # 4. 分析检测正确率
    correct_detections = sum(1 for stat in per_image_stats if stat["检测结果是否正确"])
    if len(per_image_stats) > 0:
        accuracy = correct_detections / len(per_image_stats) * 100
        detection_stats["检测正确率"] = f"{accuracy:.2f}%"
    
    # 按类别统计正确率
    accuracy_by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for stat in per_image_stats:
        actual_class = stat["实际类别"]
        if actual_class:
            accuracy_by_class[actual_class]["total"] += 1
            if stat["检测结果是否正确"]:
                accuracy_by_class[actual_class]["correct"] += 1
    
    detection_stats["各类别正确率"] = {}
    for cls, counts in accuracy_by_class.items():
        if counts["total"] > 0:
            cls_accuracy = counts["correct"] / counts["total"] * 100
            detection_stats["各类别正确率"][cls] = f"{cls_accuracy:.2f}%"
    
    # 5. 输出统计报告
    # 保存详细的图片级别统计到CSV
    df = pd.DataFrame(per_image_stats)
    csv_detail_path = os.path.join(output_dir, 'detection_details.csv')
    df.to_csv(csv_detail_path, index=False, encoding='utf-8')
    
    # 保存详细的统计报告
    detailed_stats_path = os.path.join(output_dir, 'detection_analysis.txt')
    with open(detailed_stats_path, 'w', encoding='utf-8') as f:
        f.write("玻璃瓶检测分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 基本统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"总图片数: {detection_stats['总图片数']}\n")
        f.write(f"检测出类别数: {detection_stats['检测出类别数']}\n")
        if "检测正确率" in detection_stats:
            f.write(f"总体检测正确率: {detection_stats['检测正确率']}\n")
        f.write("\n")
        
        f.write("2. 各类别检测统计\n")
        f.write("-" * 30 + "\n")
        for cls_name, count in detection_stats["类别检测统计"].items():
            f.write(f"  {cls_name}: {count} 个实例\n")
        f.write("\n")
        
        f.write("3. 各类别正确率\n")
        f.write("-" * 30 + "\n")
        if "各类别正确率" in detection_stats:
            for cls_name, accuracy in detection_stats["各类别正确率"].items():
                f.write(f"  {cls_name}: {accuracy}\n")
        f.write("\n")
        
        f.write("4. 各类别图片数量\n")
        f.write("-" * 30 + "\n")
        for bottle_type, count in images_by_type_counts.items():
            f.write(f"  {bottle_type}: {count} 张图片\n")
        f.write("\n")
        
        f.write("5. 预期统计\n")
        f.write("-" * 30 + "\n")
        f.write("各类别预期数量（基于目录计数）:\n")
        for cls_name, count in detection_stats["预期类别统计"].items():
            f.write(f"  {cls_name}: {count} 张图片\n")
        f.write("\n")
        
        f.write("6. 误差分析\n")
        f.write("-" * 30 + "\n")
        for cls_name, stats in detection_stats["类别误差分析"].items():
            f.write(f"  {cls_name}:\n")
            f.write(f"    预期数量: {stats['预期数量']}\n")
            f.write(f"    检测数量: {stats['检测数量']}\n")
            f.write(f"    误差: {stats['误差']} ({stats['误差率']})\n")
        f.write("\n")
        
        if "瓶口与瓶身比例" in detection_stats:
            f.write("7. 瓶口与瓶身比例分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"  瓶口与瓶身比例: {detection_stats['瓶口与瓶身比例']}\n")
            if "瓶口过多分析" in detection_stats:
                f.write(f"  {detection_stats['瓶口过多分析']}\n")
            elif "瓶口不足分析" in detection_stats:
                f.write(f"  {detection_stats['瓶口不足分析']}\n")
            f.write("\n")
        
        f.write("8. 缺失分析\n")
        f.write("-" * 30 + "\n")
        f.write(f"  未检测到任何物体的图片数: {len(detection_stats['缺失分析']['未检测到任何物体的图片'])}\n")
        f.write(f"  仅检测到瓶口的图片数: {len(detection_stats['缺失分析']['仅检测到瓶口的图片'])}\n")
        f.write(f"  仅检测到瓶身的图片数: {len(detection_stats['缺失分析']['仅检测到瓶身的图片'])}\n")
        f.write(f"  检测到多个瓶子的图片数: {len(detection_stats['缺失分析']['检测到多个瓶子的图片'])}\n\n")
        
        # 列出部分有问题的图片
        for issue_type, img_paths in detection_stats["缺失分析"].items():
            if img_paths:
                f.write(f"  {issue_type} (最多显示10个例子):\n")
                for img_path in img_paths[:min(10, len(img_paths))]:
                    f.write(f"    - {img_path}\n")
                f.write("\n")
    
    # 6. 生成可视化图表
    # 预期vs检测数量对比图 (仅瓶子类型)
    bottle_classes = list(bottle_types.values())
    plt.figure(figsize=(12, 8))
    
    # 设置条形图数据
    expected_vals = [expected_counts.get(cls, 0) for cls in bottle_classes]
    detected_vals = [class_counts.get(cls, 0) for cls in bottle_classes]
    
    # 设置条形图位置
    x = np.arange(len(bottle_classes))
    width = 0.35
    
    # 绘制条形图
    plt.bar(x - width/2, expected_vals, width, label='图片数量')
    plt.bar(x + width/2, detected_vals, width, label='检测数量')
    
    # 添加标签和标题
    plt.xlabel('瓶子类型')
    plt.ylabel('数量')
    plt.title('各类别预期数量vs检测数量')
    plt.xticks(x, bottle_classes, rotation=45)
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    comparison_chart_path = os.path.join(output_dir, 'class_comparison.png')
    plt.savefig(comparison_chart_path)
    plt.close()
    
    # 瓶口与瓶身比例饼图
    plt.figure(figsize=(10, 10))
    labels = ['瓶口', '瓶身']
    sizes = [class_counts.get("瓶口", 0), bottle_types_total]
    explode = (0.1, 0)  # 突出瓶口部分
    
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # 保证饼图是圆形的
    plt.title('瓶口与瓶身检测比例')
    
    # 保存图表
    ratio_chart_path = os.path.join(output_dir, 'mouth_body_ratio.png')
    plt.savefig(ratio_chart_path)
    plt.close()
    
    # 各类别检测正确率条形图
    if "各类别正确率" in detection_stats:
        plt.figure(figsize=(12, 6))
        classes = list(detection_stats["各类别正确率"].keys())
        accuracy_vals = [float(acc.strip('%')) for acc in detection_stats["各类别正确率"].values()]
        
        # 绘制条形图
        bars = plt.bar(classes, accuracy_vals, color='skyblue')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 添加标签和标题
        plt.xlabel('瓶子类型')
        plt.ylabel('正确率 (%)')
        plt.title('各类别检测正确率')
        plt.ylim(0, 110)  # 为了留出标签的空间
        
        # 保存图表
        plt.tight_layout()
        accuracy_chart_path = os.path.join(output_dir, 'class_accuracy.png')
        plt.savefig(accuracy_chart_path)
        plt.close()
    
    # 缺失分析条形图
    plt.figure(figsize=(12, 6))
    
    # 设置条形图数据
    issues = ['未检测到物体', '仅检测到瓶口', '仅检测到瓶身', '检测到多个瓶子']
    issue_counts = [
        len(detection_stats['缺失分析']['未检测到任何物体的图片']),
        len(detection_stats['缺失分析']['仅检测到瓶口的图片']),
        len(detection_stats['缺失分析']['仅检测到瓶身的图片']),
        len(detection_stats['缺失分析']['检测到多个瓶子的图片'])
    ]
    
    # 绘制条形图
    plt.bar(issues, issue_counts, color=['red', 'orange', 'yellow', 'green'])
    
    # 添加数据标签
    for i, v in enumerate(issue_counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    # 添加标签和标题
    plt.xlabel('问题类型')
    plt.ylabel('图片数量')
    plt.title('检测问题分析')
    
    # 保存图表
    plt.tight_layout()
    issues_chart_path = os.path.join(output_dir, 'detection_issues.png')
    plt.savefig(issues_chart_path)
    plt.close()
    
    print(f"分析报告已保存到: {detailed_stats_path}")
    print(f"预期vs检测数量对比图保存到: {comparison_chart_path}")
    print(f"瓶口与瓶身比例图保存到: {ratio_chart_path}")
    if "各类别正确率" in detection_stats:
        print(f"各类别检测正确率图保存到: {accuracy_chart_path}")
    print(f"检测问题分析图保存到: {issues_chart_path}")
    print(f"详细分析结果已保存到: {csv_detail_path}")
    
    return detection_stats

def detect_bottles(model_path, data_dir, output_dir, conf=0.5, iou=0.45, exclude_dirs=None):
    """
    使用YOLO模型检测目录中所有图片中的玻璃瓶
    
    参数:
        model_path: 模型路径
        data_dir: 数据目录
        output_dir: 输出目录
        conf: 置信度阈值
        iou: NMS的IoU阈值
        exclude_dirs: 需要排除的目录列表
    """
    # 创建检测器
    print(f"正在加载模型: {model_path}")
    detector = BottleDetector(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建保存检测图像的目录
    detection_images_dir = os.path.join(output_dir, 'detection_images')
    os.makedirs(detection_images_dir, exist_ok=True)
    
    # 创建保存错误检测图像的目录
    error_images_dir = os.path.join(output_dir, 'error_images')
    os.makedirs(error_images_dir, exist_ok=True)
    
    # 设置默认排除目录
    if exclude_dirs is None:
        exclude_dirs = ["mini"]  # 默认排除mini目录（训练集）
    
    # 查找所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    excluded_files = []
    
    print(f"正在查找图片文件...")
    for root, dirs, files in os.walk(data_dir):
        # 检查当前目录是否需要排除
        should_exclude = False
        for exclude_dir in exclude_dirs:
            # 判断当前路径是否包含需要排除的目录名
            if exclude_dir in root.split(os.sep):
                should_exclude = True
                break
        
        if should_exclude:
            # 如果需要排除当前目录，记录被排除的图片
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    excluded_path = os.path.join(root, file)
                    excluded_files.append(excluded_path)
            continue
        
        # 如果不需要排除，添加到要处理的图片列表
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片用于检测")
    if excluded_files:
        print(f"排除了 {len(excluded_files)} 张图片（来自训练集或指定排除的目录）")
    
    # 检测结果统计
    class_counts = {}
    detection_times = []
    detection_results = []  # 存储每张图片的检测结果
    
    # 错误检测图片列表
    misclassified_images = []  # 分类错误的图片
    undetected_images = []     # 未检测到物体的图片
    partial_detection_images = []  # 仅检测到瓶口或瓶身的图片
    
    # 创建CSV文件记录结果
    csv_path = os.path.join(output_dir, 'detection_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("图片路径,检测到的类别数量,检测到的实例数量,检测耗时(秒),检测结果是否正确\n")
    
    # 确定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 处理每张图片
    for img_path in tqdm(image_files, desc="处理图片"):
        try:
            # 记录时间
            start_time = time.time()
            
            # 提取图片文件名（不带路径和扩展名）
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # 确定图片的真实类别（基于文件路径）
            actual_class = None
            bottle_types = {
                "水滴形截面的玻璃瓶": "水滴形截面的玻璃瓶",
                "圆形截面的玻璃瓶": "圆形截面的玻璃瓶",
                "椭圆形截面的玻璃瓶": "椭圆形截面的玻璃瓶"
            }
            
            # 获取目录路径和名称
            dir_path = os.path.dirname(img_path)
            dir_name = os.path.basename(dir_path)
            
            # 检查目录名称是否匹配某个瓶子类型
            for bottle_type in bottle_types.values():
                if dir_name == bottle_type:
                    actual_class = bottle_type
                    break
            
            # 使用bottle_detector的predict方法
            results = detector.predict(img_path)
            
            # 计算处理时间
            process_time = time.time() - start_time
            detection_times.append(process_time)
            
            # 获取第一个结果
            if results and len(results) > 0:
                result = results[0]
                
                # 保存带检测框的图像
                save_path = os.path.join(detection_images_dir, f"{img_basename}_detection.jpg")
                save_detection_visualization(img_path, result, save_path)
                
                # 检查是否正确检测
                is_correct = False
                has_bottle = False
                has_mouth = False
                detected_bottle_class = None
                
                # 检查检测结果中的类别
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    classes = []
                    
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = result.names[cls_id] if cls_id in result.names else f"类别{cls_id}"
                        classes.append(cls_name)
                        
                        if cls_name == "瓶口":
                            has_mouth = True
                        elif cls_name in bottle_types.values():
                            has_bottle = True
                            detected_bottle_class = cls_name
                        
                        # 统计类别数量
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                        else:
                            class_counts[cls_name] = 1
                    
                    # 检查是否正确分类
                    if actual_class and detected_bottle_class == actual_class:
                        is_correct = True
                    # 如果只检测到瓶口，但确实是椭圆形截面的玻璃瓶，也标记为正确
                    elif has_mouth and not has_bottle and actual_class == "椭圆形截面的玻璃瓶":
                        is_correct = True
                        # 在这种情况下，我们假设检测到的是椭圆形截面的玻璃瓶
                        detected_bottle_class = "椭圆形截面的玻璃瓶"
                
                # 记录检测结果及错误分析
                detection_result = {
                    "image_path": img_path,
                    "result": result,
                    "process_time": process_time,
                    "actual_class": actual_class,
                    "detected_classes": classes if 'classes' in locals() else [],
                    "is_correct": is_correct,
                    "has_bottle": has_bottle,
                    "has_mouth": has_mouth
                }
                
                detection_results.append(detection_result)
                
                # 记录错误类型
                if not is_correct:
                    # 保存错误检测图像
                    error_save_path = os.path.join(error_images_dir, f"{img_basename}_error.jpg")
                    save_detection_visualization(img_path, result, error_save_path)
                    
                    if not has_bottle and not has_mouth:
                        undetected_images.append({
                            "image_path": img_path,
                            "actual_class": actual_class,
                            "error_type": "未检测到任何物体"
                        })
                    elif not has_bottle and has_mouth:
                        partial_detection_images.append({
                            "image_path": img_path,
                            "actual_class": actual_class,
                            "error_type": "仅检测到瓶口"
                        })
                    elif has_bottle and not has_mouth:
                        partial_detection_images.append({
                            "image_path": img_path,
                            "actual_class": actual_class,
                            "error_type": "仅检测到瓶身"
                        })
                    elif detected_bottle_class != actual_class:
                        misclassified_images.append({
                            "image_path": img_path,
                            "actual_class": actual_class,
                            "detected_class": detected_bottle_class,
                            "error_type": "瓶子类型错误"
                        })
                
                # 保存到CSV
                with open(csv_path, 'a', encoding='utf-8') as f:
                    rel_path = os.path.relpath(img_path, data_dir)
                    boxes_count = len(result.boxes) if hasattr(result, 'boxes') else 0
                    classes_count = len(set(classes)) if 'classes' in locals() and classes else 0
                    f.write(f"{rel_path},{classes_count},{boxes_count},{process_time:.4f},{is_correct}\n")
            else:
                # 检测结果为空
                undetected_images.append({
                    "image_path": img_path,
                    "actual_class": actual_class,
                    "error_type": "未检测到任何物体"
                })
                
                detection_results.append({
                    "image_path": img_path,
                    "result": None,
                    "process_time": process_time,
                    "actual_class": actual_class,
                    "detected_classes": [],
                    "is_correct": False,
                    "has_bottle": False,
                    "has_mouth": False
                })
                
                # 保存到CSV
                with open(csv_path, 'a', encoding='utf-8') as f:
                    rel_path = os.path.relpath(img_path, data_dir)
                    f.write(f"{rel_path},0,0,{process_time:.4f},False\n")
                
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            with open(csv_path, 'a', encoding='utf-8') as f:
                rel_path = os.path.relpath(img_path, data_dir)
                f.write(f"{rel_path},错误,错误,0,False\n")
    
    # 保存错误检测图片列表
    error_lists_dir = os.path.join(output_dir, 'error_lists')
    os.makedirs(error_lists_dir, exist_ok=True)
    
    # 保存未检测到物体的图片列表
    undetected_path = os.path.join(error_lists_dir, 'undetected_images.txt')
    with open(undetected_path, 'w', encoding='utf-8') as f:
        f.write("未检测到任何物体的图片列表:\n")
        f.write("=" * 50 + "\n\n")
        for item in undetected_images:
            rel_path = os.path.relpath(item["image_path"], data_dir)
            actual_class = item["actual_class"] if item["actual_class"] else "未知"
            f.write(f"{rel_path} | 实际类别: {actual_class}\n")
    
    # 保存部分检测的图片列表
    partial_path = os.path.join(error_lists_dir, 'partial_detection_images.txt')
    with open(partial_path, 'w', encoding='utf-8') as f:
        f.write("部分检测的图片列表（仅检测到瓶口或瓶身）:\n")
        f.write("=" * 50 + "\n\n")
        for item in partial_detection_images:
            rel_path = os.path.relpath(item["image_path"], data_dir)
            actual_class = item["actual_class"] if item["actual_class"] else "未知"
            error_type = item["error_type"]
            f.write(f"{rel_path} | 实际类别: {actual_class} | 错误类型: {error_type}\n")
    
    # 保存分类错误的图片列表
    misclassified_path = os.path.join(error_lists_dir, 'misclassified_images.txt')
    with open(misclassified_path, 'w', encoding='utf-8') as f:
        f.write("分类错误的图片列表:\n")
        f.write("=" * 50 + "\n\n")
        for item in misclassified_images:
            rel_path = os.path.relpath(item["image_path"], data_dir)
            actual_class = item["actual_class"] if item["actual_class"] else "未知"
            detected_class = item["detected_class"] if item["detected_class"] else "未知"
            f.write(f"{rel_path} | 实际类别: {actual_class} | 检测类别: {detected_class}\n")
    
    # 分析检测结果，生成详细统计报告
    detection_stats = analyze_detection_results(
        data_dir=data_dir,
        image_files=image_files,
        detection_results=detection_results,
        class_counts=class_counts,
        output_dir=output_dir
    )
    
    # 生成基本统计报告
    stats_path = os.path.join(output_dir, 'detection_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("玻璃瓶检测统计报告\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"总图片数: {len(image_files)}\n")
        if detection_times:
            f.write(f"平均检测时间: {sum(detection_times)/len(detection_times):.4f} 秒\n\n")
        else:
            f.write("平均检测时间: 0.0000 秒\n\n")
        
        f.write("各类别检测统计:\n")
        for cls_name, count in class_counts.items():
            f.write(f"  {cls_name}: {count} 个实例\n")
        
        # 添加错误检测统计
        f.write("\n错误检测统计:\n")
        f.write(f"  未检测到物体的图片数: {len(undetected_images)}\n")
        f.write(f"  部分检测的图片数: {len(partial_detection_images)}\n")
        f.write(f"  分类错误的图片数: {len(misclassified_images)}\n")
    
    # 生成可视化统计图表
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('各类别检测数量统计')
    plt.xlabel('类别')
    plt.ylabel('检测到的实例数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(chart_path)
    plt.close()
    
    # 生成错误类型分布图
    error_types = ['未检测到物体', '部分检测', '分类错误']
    error_counts = [len(undetected_images), len(partial_detection_images), len(misclassified_images)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(error_types, error_counts, color=['red', 'orange', 'yellow'])
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom')
    
    plt.title('错误类型分布')
    plt.xlabel('错误类型')
    plt.ylabel('图片数量')
    plt.tight_layout()
    
    # 保存错误分布图
    error_chart_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_chart_path)
    plt.close()
    
    print(f"基本统计报告已保存到: {stats_path}")
    print(f"类别分布图已保存到: {chart_path}")
    print(f"错误类型分布图已保存到: {error_chart_path}")
    print(f"错误图片列表已保存到: {error_lists_dir}")
    print(f"检测结果图像已保存到: {detection_images_dir}")
    print(f"错误检测图像已保存到: {error_images_dir}")
    print(f"检测结果已保存到: {output_dir}")
    if excluded_files:
        print(f"注意: 已排除训练集(mini)和指定目录中的 {len(excluded_files)} 张图片")

if __name__ == "__main__":
    # 配置参数
    model_path = "/Users/eric/Desktop/Python/CV/run_93/best.pt"
    data_dir = "/Users/eric/Desktop/Python/CV/检测 玻璃瓶"
    output_dir = "/Users/eric/Desktop/Python/CV/detection_results"
    conf_threshold = 0.5
    iou_threshold = 0.45  # 添加IoU阈值参数
    exclude_dirs = ["mini"]  # 排除mini目录（训练集）
    
    # 询问用户是否要调整参数
    print("玻璃瓶检测程序")
    print("-" * 30)
    
    use_default = input("是否使用默认参数? (y/n，默认y): ").strip().lower() != 'n'
    
    if not use_default:
        model_path = input(f"请输入模型路径 (默认: {model_path}): ").strip() or model_path
        data_dir = input(f"请输入数据目录 (默认: {data_dir}): ").strip() or data_dir
        output_dir = input(f"请输入输出目录 (默认: {output_dir}): ").strip() or output_dir
        
        conf_input = input(f"请输入置信度阈值 (0-1，默认: {conf_threshold}): ").strip()
        if conf_input:
            try:
                conf_threshold = float(conf_input)
                if conf_threshold < 0 or conf_threshold > 1:
                    print(f"置信度必须在0-1之间，使用默认值: {conf_threshold}")
                    conf_threshold = 0.5
            except ValueError:
                print(f"无效的置信度值，使用默认值: {conf_threshold}")
        
        iou_input = input(f"请输入IoU阈值 (0-1，默认: {iou_threshold}): ").strip()
        if iou_input:
            try:
                iou_threshold = float(iou_input)
                if iou_threshold < 0 or iou_threshold > 1:
                    print(f"IoU阈值必须在0-1之间，使用默认值: {iou_threshold}")
                    iou_threshold = 0.45
            except ValueError:
                print(f"无效的IoU阈值，使用默认值: {iou_threshold}")
        
        exclude_input = input(f"请输入要排除的目录，用逗号分隔 (默认: mini): ").strip()
        if exclude_input:
            exclude_dirs = [dir.strip() for dir in exclude_input.split(',')]
    
    # 运行检测
    print("\n开始检测...")
    detect_bottles(model_path, data_dir, output_dir, conf_threshold, iou_threshold, exclude_dirs)