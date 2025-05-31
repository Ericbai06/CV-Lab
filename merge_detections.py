import pandas as pd
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl

# 配置中文字体
def setup_chinese_font():
    """配置中文字体支持"""
    # 尝试设置中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'NSimSun',
        'FangSong',
        'KaiTi'
    ]
    
    # 设置字体
    for font in chinese_fonts:
        try:
            plt.rcParams['font.family'] = font
            # 测试字体是否可用
            fig = plt.figure()
            plt.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print(f"使用字体: {font}")
            return
        except:
            continue
    
    # 如果没有找到合适的中文字体，使用默认字体
    print("警告: 未找到合适的中文字体，使用默认字体")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 在程序开始时设置字体
setup_chinese_font()

def extract_frames(video_path, start_frame, end_frame, num_frames=6):
    """
    从视频中提取指定范围的帧
    Args:
        video_path: 视频文件路径
        start_frame: 起始帧
        end_frame: 结束帧
        num_frames: 要提取的帧数，默认7帧
    Returns:
        提取的帧列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 计算采样间隔
    total_frames = end_frame - start_frame
    if total_frames > 0:
        interval = total_frames // (num_frames - 1) if num_frames > 1 else 1
        frame_indices = [start_frame + i * interval for i in range(num_frames)]
        frame_indices[-1] = min(frame_indices[-1], end_frame)  # 确保不超过结束帧
    else:
        frame_indices = [start_frame]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def get_video_id(video_path):
    """从视频路径中提取视频ID"""
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    return video_id

def visualize_detections(video_path, df, output_dir="detection_visualizations"):
    """
    可视化检测结果
    Args:
        video_path: 视频文件路径
        df: 包含检测结果的DataFrame
        output_dir: 输出目录
    """
    # 创建基于视频ID的输出目录
    video_id = get_video_id(video_path)
    output_dir = os.path.join(output_dir, f"video_{video_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个检测结果创建可视化
    for idx, row in df.iterrows():
        # 提取帧
        frames = extract_frames(video_path, int(row['start_frame']), int(row['end_frame']))
        
        # 创建图形 - 2行4列布局（3列图片+1列信息）
        fig = plt.figure(figsize=(20, 10))  # 调整图形大小
        gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.3], height_ratios=[1, 1])
        
        # 显示帧 - 2*3排列
        for i, frame in enumerate(frames):
            row_idx = i // 3  # 确定行号
            col_idx = i % 3   # 确定列号
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(frame)
            ax.axis('off')
            
            # 设置标题
            if i == 0:
                ax.set_title(f"Start Frame: {int(row['start_frame'])}")
            elif i == len(frames) - 1:
                ax.set_title(f"End Frame: {int(row['end_frame'])}")
            else:
                ax.set_title(f"Frame: {int(row['start_frame']) + i * ((int(row['end_frame']) - int(row['start_frame'])) // (len(frames)-1))}")
        
        # 添加信息面板 - 跨越两行
        info_ax = fig.add_subplot(gs[:, -1])
        info_ax.axis('off')
        info_text = (
            f"Detection #{idx+1}\n"
            f"Time: {format_time(row['start_time'])} - {format_time(row['end_time'])}\n"
            f"Duration: {format_time(row['duration'])}\n"
            f"Confidence: {row['confidence']:.4f}\n"
            f"Frames: {int(row['start_frame'])} - {int(row['end_frame'])}"
        )
        info_ax.text(0.1, 0.5, info_text, va='center', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_file = os.path.join(output_dir, f"detection_{idx+1}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for video {video_id}: {output_file}")

def merge_overlapping_detections(df, time_threshold=0.5):
    """
    合并重叠的检测结果
    Args:
        df: 包含检测结果的DataFrame
        time_threshold: 合并的时间阈值（秒）
    Returns:
        合并后的DataFrame
    """
    # 按开始时间排序
    df = df.sort_values('start_time')
    
    # 初始化结果列表
    merged_results = []
    current_result = None
    
    for _, row in df.iterrows():
        if current_result is None:
            # 第一个结果
            current_result = {
                'start_frame': row['start_frame'],
                'end_frame': row['end_frame'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'confidence': row['confidence'],
                'is_violence': row['is_violence']
            }
        else:
            # 检查是否重叠
            time_gap = row['start_time'] - current_result['end_time']
            
            if time_gap <= time_threshold:
                # 合并重叠的结果
                current_result['end_frame'] = max(current_result['end_frame'], row['end_frame'])
                current_result['end_time'] = max(current_result['end_time'], row['end_time'])
                current_result['confidence'] = max(current_result['confidence'], row['confidence'])
            else:
                # 保存当前结果并开始新的结果
                merged_results.append(current_result)
                current_result = {
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'confidence': row['confidence'],
                    'is_violence': row['is_violence']
                }
    
    # 添加最后一个结果
    if current_result is not None:
        merged_results.append(current_result)
    
    # 转换为DataFrame
    merged_df = pd.DataFrame(merged_results)
    
    # 计算持续时间
    merged_df['duration'] = merged_df['end_time'] - merged_df['start_time']
    
    return merged_df

def format_time(seconds):
    """将秒数转换为分秒格式"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def format_csv_time(df):
    """格式化DataFrame中的时间列"""
    # 创建新的DataFrame用于显示
    display_df = df.copy()
    
    # 格式化时间列
    display_df['start_time'] = display_df['start_time'].apply(format_time)
    display_df['end_time'] = display_df['end_time'].apply(format_time)
    display_df['duration'] = display_df['duration'].apply(format_time)
    
    return display_df

def main():
    # 读取检测结果
    input_file = "/root/autodl-tmp/CV/detection_results/1_20250529_234139_windows.csv"
    video_path = "/root/autodl-tmp/CV/1.mp4"  # 视频文件路径
    df = pd.read_csv(input_file)
    
    # 只保留置信度超过阈值的检测结果
    df = df[df['confidence'] > 0.9]
    
    # 合并重叠的检测结果
    merged_df = merge_overlapping_detections(df)
    
    # 打印合并前后的统计信息
    print(f"\n原始检测结果数量: {len(df)}")
    print(f"合并后检测结果数量: {len(merged_df)}")
    
    # 打印合并后的结果
    print("\n合并后的检测结果:")
    display_df = format_csv_time(merged_df)
    for _, row in display_df.iterrows():
        print(f"\n时间段: {row['start_time']} - {row['end_time']}")
        print(f"持续时间: {row['duration']}")
        print(f"置信度: {row['confidence']:.4f}")
        print(f"帧范围: {int(row['start_frame'])} - {int(row['end_frame'])}")
    
    # 保存合并后的结果（包含原始时间数据）
    output_file = input_file.replace('_windows.csv', '_merged.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"\n合并后的结果已保存到: {output_file}")
    
    # 保存格式化后的结果
    formatted_file = input_file.replace('_windows.csv', '_merged_formatted.csv')
    display_df.to_csv(formatted_file, index=False)
    print(f"格式化后的结果已保存到: {formatted_file}")
    
    # 创建可视化
    print("\n开始创建可视化...")
    visualize_detections(video_path, merged_df)
    print("可视化完成！")

if __name__ == "__main__":
    main() 