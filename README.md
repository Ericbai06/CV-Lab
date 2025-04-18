# 玻璃瓶检测系统

基于YOLOv8的智能玻璃瓶检测系统，用于高效识别图像中的玻璃瓶及其瓶口，支持多种瓶型分类。

## 项目概述

本系统采用改进的YOLOv8目标检测算法，结合自定义后处理方法，实现对玻璃瓶的精确检测与分类。系统可识别三种不同截面类型的玻璃瓶（圆形、椭圆形、水滴形）及其瓶口位置，并提供图形化界面支持交互式检测。

## 性能指标

- **总测试图片数**：949张
- **平均检测速度**：0.038秒/图
- **瓶身分类正确率**：97.15%（瓶身分类错误15张，未识别12张）
- **瓶口检测正确率**：99.47%（未检测出瓶口5张）

## 系统特点

- **高准确率**：瓶身和瓶口检测准确率均超过97%
- **实时性能**：平均每张图片检测时间仅0.038秒，约26FPS
- **多瓶型识别**：支持圆形、椭圆形和水滴形三种截面玻璃瓶的识别
- **用户友好界面**：提供直观的图形用户界面，便于操作和结果查看
- **智能后处理**：自动推断缺失的瓶口或瓶身，增强检测鲁棒性

## 安装要求

- Python 3.8+
- PyTorch 1.7+
- Ultralytics YOLOv8
- OpenCV
- Tkinter (GUI界面)
- Matplotlib
- NumPy

## 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-username/glass-bottle-detection.git
cd glass-bottle-detection

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型（如果没有自动下载）
# 项目默认使用 yolov8x.pt 或自定义训练的模型
```

## 使用方法

### GUI界面

```bash
python 玻璃瓶检测.py
```

1. 点击"打开图像"选择要检测的图片
2. 点击"运行检测"进行瓶子检测
3. 点击"保存结果"保存检测结果图像

### 批量测试

```bash
python bottle_detection_test.py --model yolov8x.pt --data_dir "检测 玻璃瓶" --output_dir "detection_results"
```

## 检测类别

系统能够识别以下类别，定义在`data.yaml`文件中：

- **类别0**：圆形截面的玻璃瓶
- **类别1**：瓶口
- **类别2**：椭圆形截面的玻璃瓶
- **类别3**：水滴形截面的玻璃瓶

## 项目结构

- `bottle_detector.py` - 核心检测引擎和GUI界面实现
- `bottle_detection_utils.py` - 检测算法工具函数
- `bottle_detection_test.py` - 批量测试与评估脚本
- `bottle_detection_visualization.py` - 结果可视化工具
- `data.yaml` - 数据集配置文件
- `yolov8x.pt` - 预训练模型

## 算法原理

本系统基于YOLOv8单阶段目标检测算法，并增加了自定义后处理步骤以提高检测质量：

1. 使用YOLOv8模型进行基础目标检测
2. 应用自定义后处理规则处理重叠检测和互补类别（如瓶身和瓶口）
3. 实现瓶身和瓶口智能关联，当只检测到其中一个时自动推断另一个

## 示例图像

![多瓶检测示例](多瓶检测.png)

