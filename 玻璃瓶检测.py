from ultralytics import YOLO
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import matplotlib
import torch

# --- Matplotlib 配置 (全局) ---
# 根据字体检查结果，优先使用找到的 Songti SC 和 Arial Unicode MS
matplotlib.rcParams['font.sans-serif'] = ['Songti SC',]
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 全局常量 ---
# 使用绝对路径更可靠
dataset_dir = "/Users/eric/Desktop/Python/CV/检测 玻璃瓶"

# --- 控制开关 ---
SKIP_TRAINING = True  # 设置为 True 跳过训练，直接加载最新 best.pt 进行验证和预测

def main():
    """主执行函数，包含模型加载、训练、验证和预测逻辑"""
    print("--- 开始执行主程序 ---")

    # --- 加载模型 ---
    # 初始加载模型结构，权重将在训练或加载 best.pt 后确定
    print("加载模型结构 yolo11n.pt (权重将在后续步骤确定)...")
    model = YOLO('yolo11n.pt')
    best_model = None # 初始化 best_model 变量
    results = None # 初始化 results，如果跳过训练则保持 None
    training_successful = False # 初始化 training_successful

    # --- 训练模型 (根据 SKIP_TRAINING 开关决定是否执行) ---
    if not SKIP_TRAINING:
        print("\n--- 开始训练模型 --- (SKIP_TRAINING is False)")
        try:
            print("开始训练 (epochs=50, device='MPS', imgsz=640, batch=4, patience=50, augment=True, dropout=0.2, mosaic=1.0, resume=False)...")
            results = model.train(
                data='data.yaml', 
                epochs=50,
                device='MPS',
                imgsz=640,
                batch=4,
                patience=50,
                augment=True,
                dropout=0.2,
                mosaic=1.0,
                resume=False
            )
            print("--- 训练结束 ---")
            training_successful = True # 标记训练成功
        except FileNotFoundError as e:
            print(f"训练过程中发生错误: 找不到文件或目录 - {e}")
            print("请确保 'data.yaml' 文件存在，并且其中指定的路径和数据集结构正确。")
        except Exception as e:
            print(f"训练过程中发生未知错误: {e}")
            # 可以在这里添加更详细的错误处理或日志记录
    else:
        print("\n--- 跳过训练模型 --- (SKIP_TRAINING is True)")

    # 只有在 未跳过训练 且 训练失败 的情况下才退出
    if not SKIP_TRAINING and not training_successful:
        print("\n训练未成功完成，跳过验证和预测步骤。")
        print("--- 主程序提前结束 ---")
        return # 退出 main 函数

    # --- 验证最佳模型并可视化各类别 mAP ---
    print("\n--- 开始验证最佳模型 ---")
    save_dir = None # 初始化 save_dir
    try:
        # 尝试从训练结果获取保存目录
        if hasattr(results, 'save_dir') and results.save_dir:
            save_dir = results.save_dir
        elif isinstance(results, str) and os.path.isdir(results): # 如果 results 直接是路径字符串
             save_dir = results
        else:
            # 如果无法从 results 获取，尝试查找最新的 train 目录
            try:
                runs_dir = 'runs/detect'
                all_train_dirs = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))])
                if all_train_dirs:
                    save_dir = all_train_dirs[-1] # 获取最新的训练目录
                    print(f"自动检测到最新的训练目录：{save_dir}")
                else:
                    raise FileNotFoundError("在 runs/detect/ 中未找到 train* 目录")
            except Exception as find_dir_e:
                save_dir = "runs/detect/train" # 最终回退默认值
                print(f"警告：无法自动确定保存目录 ({find_dir_e})，将使用默认路径：{save_dir}")

        best_model_path = os.path.join(save_dir, 'weights/best.pt')

        if not os.path.exists(best_model_path):
            print(f"错误：找不到最佳模型权重文件：{best_model_path}")
            print("验证步骤无法继续。")
        else:
            print(f"加载最佳模型：{best_model_path}")
            best_model = YOLO(best_model_path) # 将加载的最佳模型赋值给 best_model

            print("运行验证...")
            # 在调用 val() 之前再次强制设置字体，尝试覆盖 Ultralytics 内部可能的默认设置
            matplotlib.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'PingFang SC', 'SimHei', 'STSong']
            # 运行验证，明确指定数据集配置
            metrics = best_model.val(data='data.yaml')

            # 打印总体指标
            print(f"验证结果 mAP50-95: {metrics.box.map:.4f}")
            print(f"验证结果 mAP50:    {metrics.box.map50:.4f}")
            print(f"验证结果 mAP75:    {metrics.box.map75:.4f}")

            # 可视化各类别 mAP50-95
            maps_per_category = metrics.box.maps
            class_names = best_model.names # 从加载的模型获取类别名称

            if maps_per_category is not None and class_names:
                if not isinstance(maps_per_category, (list, np.ndarray)):
                     print("警告：验证结果中无法获取有效的各类别 mAP 数据进行可视化。")
                elif len(maps_per_category) != len(class_names):
                     print(f"警告：验证结果中的类别数量({len(class_names)})与mAP值数量({len(maps_per_category)})不匹配，无法可视化。")
                else:
                    maps_values = np.array(maps_per_category)
                    # 从模型的 names 属性获取类别名列表
                    sorted_class_names = [class_names[i] for i in range(len(maps_values))]

                    print("\n各类别 mAP50-95:")
                    for name, val in zip(sorted_class_names, maps_values):
                        print(f"  {name}: {val:.4f}")

                    # 创建条形图
                    plt.figure(figsize=(max(6, len(sorted_class_names) * 1.2), 6))
                    bars = plt.bar(sorted_class_names, maps_values, color='skyblue')
                    plt.xlabel("类别")
                    plt.ylabel("mAP50-95")
                    plt.title("各类别 mAP50-95 评估结果 (验证集)")
                    plt.ylim(0, 1.05) # 留一点顶部空间
                    plt.xticks(rotation=45, ha='right') # 旋转标签以防重叠

                    # 在条形图上显示数值
                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', va='bottom', ha='center', fontsize=9)

                    plt.tight_layout(pad=1.5) # 调整布局
                    plot_save_path = os.path.join(save_dir, "category_map_validation_plot.png") # 文件名明确是验证集结果
                    try:
                        plt.savefig(plot_save_path)
                        print(f"各类别 mAP 图表已保存至: {plot_save_path}")
                    except Exception as e:
                        print(f"保存图表时出错: {e}")
                    plt.show() # 显示图表
            else:
                 print("无法从验证结果获取各类别 mAP 数据或类别名称进行可视化。")

    except Exception as e:
        print(f"验证或可视化过程中发生错误: {e}")
        # 即使验证失败，如果模型已加载，仍尝试继续预测
        if best_model is None and os.path.exists(best_model_path): # 尝试再次加载以进行预测
             try:
                  print(f"验证失败，但尝试重新加载模型 {best_model_path} 以进行预测...")
                  best_model = YOLO(best_model_path)
             except Exception as load_err:
                  print(f"重新加载模型失败: {load_err}")

    print("--- 验证和可视化结束 ---")


    # --- 推理预测示例 ---
    print("\n--- 开始运行推理预测示例 ---")
    if best_model is not None: # 检查 best_model 是否已成功加载
        print("使用验证阶段加载的最佳模型进行预测...")
        # 尝试：在预测前重新加载模型，以确保状态干净（特别是在 MPS 上）
        if 'best_model_path' in locals() and os.path.exists(best_model_path):
            print(f"为确保状态干净，重新加载模型: {best_model_path}")
            try:
                best_model = YOLO(best_model_path) # 重新加载
            except Exception as reload_err:
                print(f"重新加载模型失败: {reload_err}, 将继续使用之前的模型实例。")
        else:
             print("警告：无法找到 best_model_path 或文件不存在，无法重新加载模型。")

        prediction_sources = []
        # 使用全局 dataset_dir
        source_base_dir = dataset_dir
        # 定义用于查找示例图片的原始类别目录 (如果需要，可以做成全局常量)
        original_categories_for_source = ["水滴形截面的玻璃瓶", "椭圆形截面的玻璃瓶", "圆形截面的玻璃瓶"]

        print("查找预测示例图片...")
        for category in original_categories_for_source:
            category_path = os.path.join(source_base_dir, category)
            try:
                # 获取目录下所有图片文件
                images_in_category = [os.path.join(category_path, f)
                                      for f in os.listdir(category_path)
                                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if images_in_category:
                    # 随机选择一张图片，而不是总是第一张
                    chosen_image = random.choice(images_in_category)
                    prediction_sources.append(chosen_image)
                    print(f"  从 '{category}' 选取: {Path(chosen_image).name}")
                else:
                    print(f"警告：在目录 {category_path} 中未找到用于预测的图片。")
            except FileNotFoundError:
                print(f"警告：找不到目录 {category_path}。")
            except Exception as e:
                print(f"查找图片时出错 ({category_path}): {e}")

        if not prediction_sources:
            print("错误：未能找到任何用于预测的示例图片，跳过预测步骤。")
        else:
            print(f"将对找到的 {len(prediction_sources)} 张图片进行预测...")
            try:
                # 使用 torch.no_grad() 包装预测调用，尝试解决 MPS 上的 tensor version counter 问题
                with torch.no_grad():
                    predict_results = best_model.predict(source=prediction_sources, save=True, show=False, conf=0.5, device='MPS') # 指定设备

                predict_save_dir = "runs/detect/predict" # 默认提示
                # 尝试获取准确的保存路径
                if predict_results and hasattr(predict_results[0], 'save_dir') and predict_results[0].save_dir:
                    predict_save_dir = predict_results[0].save_dir

                print(f"预测完成。带有边界框的结果图像已保存到目录: '{predict_save_dir}'")

                # 可选：打印每个预测图片的详细检测结果
                # print("\n详细预测结果:")
                # for i, r in enumerate(predict_results):
                #     print(f"--- 结果 for {Path(prediction_sources[i]).name} ---")
                #     r.print() # 使用 Ultralytics 内置的打印方法显示结果摘要

            except Exception as e:
                print(f"运行预测时发生错误: {e}")

    else:
        print("错误：未成功加载最佳模型 ('best_model' is None)，无法进行推理预测。")

    print("--- 推理预测结束 ---")
    print("--- 主程序执行完毕 ---")


# # --- 旧的、被注释掉的函数定义可以保留在这里或移除 ---
# # 显示部分结果示例 (注释掉)
# def show_sample_results(results_directory, num_samples=9, save_path=None):
#     # ... (函数体被注释掉) ...
#     pass # 函数体被注释掉了，加个 pass 避免语法错误


# --- 主程序入口 ---
if __name__ == "__main__":
    # 可以在这里添加命令行参数解析 (argparse) 如果需要更多配置灵活性
    main()