import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics.utils.plotting import Annotator, colors

def detect_bottle_with_neck(image_path, model, conf=0.5, device='cpu', iou=0.45):
    """
    对图像进行玻璃瓶检测，同时处理瓶颈部分的检测
    
    参数:
        image_path (str): 要检测的图像路径
        model (YOLO): 加载好的YOLO模型对象
        conf (float): 检测置信度阈值，默认0.5
        device (str): 使用的设备，如'cpu'或'cuda'（如有GPU）
        iou (float): NMS的IoU阈值，默认0.45，降低此值可减少重叠检测
        
    返回:
        list: 检测结果列表
    """
    # 运行YOLO预测
    results = model.predict(
        source=image_path,
        conf=conf,
        device=device,
        verbose=False,
        save=True,
        save_txt=False,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        visualize=False,
        augment=False,
        iou=iou  # 添加IoU阈值参数
    )
    
    # 后处理：修正椭圆形截面误检为瓶口的问题
    for i, result in enumerate(results):
        if len(result.boxes) > 0:
            # 先处理类别修正问题（将小面积椭圆形截面修正为瓶口）
            for j, box in enumerate(result.boxes):
                cls = box.cls[0].cpu().item()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 计算框的面积和宽高比
                area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                
                # 修正椭圆形截面的玻璃瓶误检为瓶口
                # 类别索引2是椭圆形截面的玻璃瓶，类别索引1是瓶口
                if cls == 2 and area < 0.1 and aspect_ratio >= 0.8 and aspect_ratio <= 1.2:
                    # 将小面积且接近正方形的椭圆形截面修正为瓶口
                    # 注意这里要直接修改YOLO结果的张量
                    box.cls[0] = torch.tensor([1.0], device=box.cls.device)
            
            # 如果同一区域有多个类别检测，只保留最高置信度的类别
            if len(result.boxes) > 1:
                # 将所有检测框转换为格式 [x1, y1, x2, y2, conf, cls]
                boxes = []
                for j, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    boxes.append([x1, y1, x2, y2, conf, cls])
                
                boxes = np.array(boxes)
                
                # 按置信度排序（降序）
                sort_idx = np.argsort(-boxes[:, 4])
                boxes = boxes[sort_idx]
                
                # 特殊规则：瓶口类别(类别索引1)优先保留
                # 其他类别(0,2,3)之间如果有重叠则只保留置信度最高的一个
                bottle_types = []  # 保存非瓶口类别的索引
                bottle_mouth = []  # 保存瓶口类别的索引
                
                # 先分类
                for j in range(len(boxes)):
                    if boxes[j, 5] == 1:  # 瓶口类别
                        bottle_mouth.append(j)
                    else:  # 瓶身类别
                        bottle_types.append(j)
                        
                # 处理瓶身类别，保留最高置信度且无明显重叠的
                kept_bottle_types = []
                for j in bottle_types:
                    should_keep = True
                    box1 = boxes[j, :4]
                    cls1 = boxes[j, 5]
                    
                    for k in kept_bottle_types:
                        box2 = boxes[k, :4]
                        iou_val = box_iou(box1, box2)
                        # 如果重叠度较高，则不保留当前框
                        if iou_val > 0.3:  # 降低IoU阈值以更严格地过滤重叠框
                            should_keep = False
                            break
                            
                    if should_keep:
                        kept_bottle_types.append(j)
                
                # 合并瓶口类别和保留的瓶身类别
                kept_indices = kept_bottle_types + bottle_mouth
                
                # 重建结果
                if len(kept_indices) < len(boxes):
                    # 需要过滤结果
                    new_boxes = result.boxes[0:0]  # 创建空的Boxes对象
                    
                    for idx in kept_indices:
                        new_boxes.append(result.boxes[sort_idx[idx]])
                    
                    # 替换原始结果
                    result.boxes = new_boxes
    
    return results

def box_iou(box1, box2):
    """计算两个框的IoU"""
    # 交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # IoU
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou
