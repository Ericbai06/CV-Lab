from ultralytics import YOLO
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import matplotlib
import torch
import argparse
from bottle_detection_utils import detect_bottle_with_neck
# --- Matplotlib 配置 (全局) ---
# 根据字体检查结果，优先使用找到的 Songti SC 和 Arial Unicode MS
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 全局常量 ---
# 使用绝对路径更可靠
dataset_dir = "检测 玻璃瓶"

# --- 控制开关 ---
SKIP_TRAINING = False  # 设置为 True 跳过训练步骤，直接使用已训练的模型进行测试

def main():
    """主执行函数，包含模型加载、训练、验证和预测逻辑"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="玻璃瓶检测程序")
    parser.add_argument("--images", nargs="+", help="要检测的图片路径，可以指定多个")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练步骤")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--max_images", type=int, default=30, help="最多处理的图片数量")
    parser.add_argument("--predict_single", type=str, help="预测单张图片的路径")
    parser.add_argument("--test", action="store_true", help="运行测试模式")
    parser.add_argument("--test_dir", type=str, help="测试目录路径")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--num_test", type=int, default=5, help="测试的图片数量")
    parser.add_argument("--device", type=str, default="MPS", help="使用的设备，如'cpu'或'cuda'")
    args = parser.parse_args()
    
    # 如果通过命令行设置了skip_training，则覆盖全局变量
    global SKIP_TRAINING
    if args.skip_training:
        SKIP_TRAINING = True
    
    # 处理单张图片预测
    if args.predict_single:
        predict_single_image(
            image_path=args.predict_single,
            model_path=args.model_path,
            conf=args.conf,
            device=args.device
        )
        return
    
    # 处理测试模式
    if args.test:
        test_prediction(
            test_dir=args.test_dir,
            model_path=args.model_path,
            conf=args.conf,
            num_images=args.num_test,
            device=args.device
        )
        return
    
    print("--- 开始执行主程序 ---")

    # --- 加载模型 ---
    # 初始加载模型结构，权重将在训练或加载 best.pt 后确定
    print("加载模型结构 yolo11n.pt (权重将在后续步骤确定)...")
    model = YOLO('yolo11n.pt')  # 使用YOLOv11n模型
    best_model = None # 初始化 best_model 变量
    results = None # 初始化 results，如果跳过训练则保持 None
    training_successful = False # 初始化 training_successful

    # --- 训练模型 (根据 SKIP_TRAINING 开关决定是否执行) ---
    if not SKIP_TRAINING:
        print("\n--- 开始训练模型 --- (SKIP_TRAINING is False)")
        try:
            print("开始训练 (epochs=100, device='cuda', imgsz=1280, batch=4, patience=50, augment=True, mosaic=1.0, resume=False)...")
            results = model.train(
                data='data.yaml', 
                epochs=150,  # 增加训练轮数
                device=args.device, # 使用命令行或默认指定的设备
                imgsz=1280,
                batch=4,
                patience=70,  # 增加早停耐心值
                augment=True,
                mosaic=1.0,
                flipud=0.5,  # 添加上下翻转增强
                fliplr=0.5,  # 添加左右翻转增强
                scale=0.5,   # 添加尺度增强
                hsv_h=0.015, # 色调增强
                hsv_s=0.7,   # 饱和度增强
                hsv_v=0.4,   # 亮度增强
                degrees=180.0, # 添加随机旋转增强，范围 +/- 90 度
                resume=False,
                overlap_mask=True,  # 使用重叠掩码提高精度
                single_cls=False,   # 多类别检测
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
                    # 添加所有图片而不是随机选择一张
                    prediction_sources.extend(images_in_category)
                    print(f"  从 '{category}' 添加了 {len(images_in_category)} 张图片")
                else:
                    print(f"警告：在目录 {category_path} 中未找到用于预测的图片。")
            except FileNotFoundError:
                print(f"警告：找不到目录 {category_path}。")
            except Exception as e:
                print(f"查找图片时出错 ({category_path}): {e}")

        # 增加检测mini目录下的图片
        mini_images_path = os.path.join(source_base_dir, "mini/images")
        try:
            if os.path.exists(mini_images_path):
                mini_images = [os.path.join(mini_images_path, f)
                               for f in os.listdir(mini_images_path)
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                prediction_sources.extend(mini_images)
                print(f"  从 'mini/images' 添加了 {len(mini_images)} 张图片")
        except Exception as e:
            print(f"查找mini目录图片时出错: {e}")
            
        # 添加用户通过命令行指定的图片
        if args.images:
            valid_images = []
            for img_path in args.images:
                if os.path.exists(img_path) and img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                    valid_images.append(img_path)
                else:
                    print(f"警告：找不到图片或格式不支持: {img_path}")
            
            if valid_images:
                prediction_sources.extend(valid_images)
                print(f"  从命令行参数添加了 {len(valid_images)} 张图片")

        # 如果图片太多，可以限制数量以避免处理时间过长
        max_images = args.max_images  # 使用命令行指定的最大图片数量
        if len(prediction_sources) > max_images:
            print(f"找到 {len(prediction_sources)} 张图片，但为避免处理时间过长，随机选择 {max_images} 张进行预测")
            random.shuffle(prediction_sources)
            prediction_sources = prediction_sources[:max_images]

        if not prediction_sources:
            print("错误：未能找到任何用于预测的示例图片，跳过预测步骤。")
        else:
            print(f"将对找到的 {len(prediction_sources)} 张图片进行预测...")
            try:
                # 使用 torch.no_grad() 包装预测调用，尝试解决 MPS 上的 tensor version counter 问题
                with torch.no_grad():
                    predict_results = []
                    for i, source in enumerate(prediction_sources):
                        print(f"正在预测第 {i+1}/{len(prediction_sources)} 张图片: {os.path.basename(source)}")
                        # 使用我们的自定义函数替代直接调用model.predict
                        result = detect_bottle_with_neck(
                            image_path=source, 
                            model=best_model, 
                            conf=args.conf,  # 使用命令行指定的置信度阈值
                            device='cpu'
                        )
                        predict_results.extend(result)
    
                predict_save_dir = "runs/detect/predict"
                # 后续处理与原来相同...

                print(f"预测完成。带有边界框的结果图像已保存到目录: '{predict_save_dir}'")

                # 可选：打印每个预测图片的详细检测结果
                print("\n详细预测结果:")
                for i, r in enumerate(predict_results):
                    print(f"--- 结果 for {Path(prediction_sources[i%len(prediction_sources)]).name} ---")
                    print(r.verbose())  # 使用verbose方法获取详细信息字符串

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

# --- 单张图片预测函数 ---
def predict_single_image(image_path, model_path=None, conf=0.5, device='MPS'):
    """
    预测单张图片
    
    参数:
        image_path (str): 图片路径
        model_path (str): 模型路径，默认为None，使用最佳模型
        conf (float): 置信度阈值
        device (str): 使用的设备
        
    返回:
        None
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到图片: {image_path}")
        return
    
    print(f"开始预测图片: {image_path}")
    
    # 加载模型
    if model_path is None or not os.path.exists(model_path):
        # 尝试查找最新的best.pt
        runs_dir = 'runs/detect'
        all_train_dirs = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                               if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))])
        if all_train_dirs:
            model_path = os.path.join(all_train_dirs[-1], 'weights/best.pt')
            print(f"自动检测到最新的训练模型：{model_path}")
        else:
            print("错误：找不到训练模型，使用基础模型yolo11n.pt")
            model_path = 'yolo11n.pt'
    
    model = YOLO(model_path)
    
    # 预测
    try:
        results = detect_bottle_with_neck(
            image_path=image_path,
            model=model,
            conf=conf,
            device=device
        )
        
        # 输出结果
        print("\n预测结果:")
        for i, r in enumerate(results):
            print(r.verbose())  # 使用verbose方法获取详细信息字符串
            
        print(f"\n预测完成。结果图像已保存到目录: 'runs/detect/predict'")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")

# --- 测试预测功能 ---
def test_prediction(test_dir=None, model_path=None, conf=0.5, num_images=5, device='MPS'):
    """
    测试预测功能
    
    参数:
        test_dir (str): 测试目录，默认为None，使用数据集目录
        model_path (str): 模型路径，默认为None，使用最佳模型
        conf (float): 置信度阈值
        num_images (int): 要测试的图片数量
        device (str): 使用的设备
        
    返回:
        None
    """
    if test_dir is None:
        test_dir = dataset_dir
    
    # 查找所有图片
    image_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"错误：在目录 {test_dir} 中未找到图片")
        return
    
    # 随机选择指定数量的图片
    if len(image_files) > num_images:
        random.shuffle(image_files)
        image_files = image_files[:num_images]
    
    print(f"将测试 {len(image_files)} 张图片...")
    
    # 加载模型
    if model_path is None or not os.path.exists(model_path):
        # 尝试查找最新的best.pt
        runs_dir = 'runs/detect'
        all_train_dirs = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                               if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))])
        if all_train_dirs:
            model_path = os.path.join(all_train_dirs[-1], 'weights/best.pt')
            print(f"自动检测到最新的训练模型：{model_path}")
        else:
            print("错误：找不到训练模型，使用基础模型yolo11n.pt")
            model_path = 'yolo11n.pt'
    
    model = YOLO(model_path)
    
    # 预测每张图片
    for i, image_path in enumerate(image_files):
        print(f"\n正在预测第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
        try:
            results = detect_bottle_with_neck(
                image_path=image_path,
                model=model,
                conf=conf,
                device=device
            )
            
            # 简要输出结果
            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                print(f"  检测到 {len(results[0].boxes)} 个目标")
                print(results[0].verbose())  # 使用verbose方法获取详细信息字符串
            else:
                print("  未检测到目标")
                
        except Exception as e:
            print(f"预测过程中发生错误: {e}")
    
    print(f"\n测试完成。结果图像已保存到目录: 'runs/detect/predict'")