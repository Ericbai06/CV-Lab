from ultralytics import YOLO
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# 加载预训练模型
model = YOLO('yolov11n.pt')

# 定义数据集路径
dataset_dir = "检测 玻璃瓶"
categories = ["水滴形截面的玻璃瓶", "椭圆形截面的玻璃瓶", "圆形截面的玻璃瓶"]

# 创建结果目录
results_dir = "检测结果"
os.makedirs(results_dir, exist_ok=True)

# 对每个类别进行测试
for category in categories:
    # 创建类别结果目录
    category_results_dir = os.path.join(results_dir, category)
    os.makedirs(category_results_dir, exist_ok=True)
    
    # 获取当前类别的图像路径
    category_dir = os.path.join(dataset_dir, category)
    image_paths = [os.path.join(category_dir, img) for img in os.listdir(category_dir) 
                  if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"正在处理 {category} 类别，共找到 {len(image_paths)} 张图像")
    
    # 对每张图像进行检测
    for img_path in image_paths:
        # 执行推理
        results = model(img_path)
        
        # 获取原始图像名称
        img_name = Path(img_path).stem
        
        # 保存带有检测结果的图像
        for r in results:
            im_with_boxes = r.plot()  # 绘制带有检测框的图像
            output_path = os.path.join(category_results_dir, f"{img_name}_result.jpg")
            cv2.imwrite(output_path, im_with_boxes)
            
            # 打印检测结果信息
            if len(r.boxes) > 0:
                print(f"图像 {img_name}: 检测到 {len(r.boxes)} 个对象")
                # 打印每个检测到的对象的类别和置信度
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = round(box.conf[0].item(), 2)
                    cls_name = model.names[cls_id]
                    print(f"  类别: {cls_name}, 置信度: {conf}")
            else:
                print(f"图像 {img_name}: 未检测到对象")

print(f"检测完成，结果已保存至 {results_dir} 目录")

# 显示部分结果示例
def show_sample_results(num_samples=3):
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(15, 10))
    
    for i, category in enumerate(categories):
        category_results_dir = os.path.join(results_dir, category)
        result_images = [f for f in os.listdir(category_results_dir) if f.endswith('_result.jpg')]
        
        for j in range(min(num_samples, len(result_images))):
            if len(result_images) > j:
                img_path = os.path.join(category_results_dir, result_images[j])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{category}\n{Path(result_images[j]).stem}")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "结果示例.png"))
    plt.show()

# 显示结果示例
show_sample_results()