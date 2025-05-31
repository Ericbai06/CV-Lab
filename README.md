# 视频暴力检测系统

基于VideoMAE的视频打斗场景检测系统，支持视频帧提取、暴力检测和结果可视化。

## 项目结构

```
violence_detection/
├── 
├── models/                 # 模型文件
├── utils/                  # 工具函数
├── data/                   # 数据文件
├── results/               # 检测结果
├── visualizations/        # 可视化结果
├── README.md             # 项目说明
└── requirements.txt      # 依赖包列表
```

## 功能特点

1. 视频暴力检测
   - 基于VideoMAE模型
   - 支持滑动窗口检测
   - 可配置的置信度阈值

2. 结果处理
   - 检测结果合并
   - 重叠片段处理
   - 时间戳格式化

3. 可视化
   - 2*3布局的帧展示
   - 检测信息面板
   - 支持多视频结果管理

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行检测：
```bash
python src/detect_violence.py --video_path <视频路径>
```

3. 处理检测结果：
```bash
python src/merge_detections.py
```

## 参数说明

- 置信度阈值：0.9
- 滑动窗口大小：16帧
- 可视化帧数：6帧
- 合并时间阈值：0.5秒

## 注意事项

1. 确保视频文件格式正确
2. 检测结果保存在results目录
3. 可视化结果保存在visualizations目录
