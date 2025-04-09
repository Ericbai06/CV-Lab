import os
import shutil
import random
from pathlib import Path
import argparse
import sys # 导入sys以更好地处理编码问题

def create_yolo_structure(source_dir, output_dir, train_ratio=0.8):
    """
    从按类别组织的源目录创建 YOLOv8 数据集结构。

    Args:
        source_dir (str): 包含类别子目录的源数据路径。
        output_dir (str): 输出 YOLO 结构的目标路径 (通常与 source_dir 相同)。
        train_ratio (float): 用于训练集的数据比例 (0.0 到 1.0)。
    """
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练集比例: {train_ratio:.0%}")

    # --- 1. 创建目标目录 ---
    train_img_dir = Path(output_dir) / 'train' / 'images'
    train_lbl_dir = Path(output_dir) / 'train' / 'labels'
    val_img_dir = Path(output_dir) / 'val' / 'images'
    val_lbl_dir = Path(output_dir) / 'val' / 'labels'

    try:
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)
        print("\n已创建/确认目标目录结构:")
        print(f"  - {train_img_dir}")
        print(f"  - {train_lbl_dir}")
        print(f"  - {val_img_dir}")
        print(f"  - {val_lbl_dir}")
    except OSError as e:
        print(f"错误：创建目录时失败 - {e}")
        return

    # --- 2. 收集源文件 ---
    all_files_by_category = {}
    try:
        # 使用 os.scandir 以更好地处理潜在的非 UTF-8 编码目录名（虽然 macOS 通常是 UTF-8）
        categories = [entry.name for entry in os.scandir(source_dir)
                      if entry.is_dir() and entry.name not in ['train', 'val'] and not entry.name.startswith('.')]

        if not categories:
            print(f"错误：在源目录 '{source_dir}' 中未找到任何有效的类别子目录。")
            print("请确保您的图片放在类似 '水滴形截面的玻璃瓶' 这样的子目录中。")
            return
        print(f"\n发现源类别目录: {categories}")

        for category in categories:
            category_path = Path(source_dir) / category
            try:
                images = [entry.name for entry in os.scandir(category_path)
                          if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')) and not entry.name.startswith('.')]
                if images:
                    all_files_by_category[category] = images
                    print(f"  - 类别 '{category}': 找到 {len(images)} 张图片")
                else:
                    print(f"  - 类别 '{category}': 未找到图片文件，已跳过。")
            except Exception as list_e:
                 print(f"  - 读取类别 '{category}' 目录时出错: {list_e}")


        if not all_files_by_category:
            print(f"错误：在所有类别子目录中均未找到图片文件。")
            return

    except FileNotFoundError:
        print(f"错误：源目录 '{source_dir}' 不存在。")
        return
    except Exception as e:
        print(f"读取源目录时出错: {e}")
        return

    # --- 3. 分割并复制文件 ---
    print(f"\n按 {train_ratio:.0%} / {1-train_ratio:.0%} 的比例分割数据并复制文件...")
    train_count = 0
    val_count = 0

    for category, images in all_files_by_category.items():
        category_source_path = Path(source_dir) / category
        if not images: continue # 如果该类别没有图片，则跳过

        random.shuffle(images) # 在类别内随机打乱
        split_index = int(len(images) * train_ratio)
        # 确保即使只有一个文件，验证集也能至少包含一个（如果可能）
        if len(images) > 1 and split_index == len(images): # 如果比例导致所有文件都在训练集
            split_index -= 1 # 至少留一个给验证集
        elif len(images) == 1 and train_ratio > 0: # 如果只有一个文件且比例大于0
             split_index = 1 # 单个文件放入训练集
        elif train_ratio == 0: # 如果比例为0
             split_index = 0 # 全部放入验证集

        train_images = images[:split_index]
        val_images = images[split_index:]

        print(f"  处理类别 '{category}': {len(train_images)} 训练, {len(val_images)} 验证")

        # 复制训练图片并创建空的标签文件
        for img_name in train_images:
            source_img_path = category_source_path / img_name
            target_img_path = train_img_dir / img_name
            # 使用 stem 获取无后缀的文件名
            target_lbl_path = train_lbl_dir / (Path(img_name).stem + '.txt')

            try:
                shutil.copy2(source_img_path, target_img_path)
                target_lbl_path.touch() # 创建空文件
                train_count += 1
            except Exception as e:
                print(f"    复制/创建训练文件时出错 ({img_name}): {e}")

        # 复制验证图片并创建空的标签文件
        for img_name in val_images:
            source_img_path = category_source_path / img_name
            target_img_path = val_img_dir / img_name
            target_lbl_path = val_lbl_dir / (Path(img_name).stem + '.txt')

            try:
                shutil.copy2(source_img_path, target_img_path)
                target_lbl_path.touch() # 创建空文件
                val_count += 1
            except Exception as e:
                print(f"    复制/创建验证文件时出错 ({img_name}): {e}")

    # --- 4. 完成与提示 ---
    print(f"\n处理完成。共处理 {train_count} 个训练文件，{val_count} 个验证文件。")
    print("\n" + "="*30 + " 重要提示 " + "="*30)
    print(f"  - 图片已复制到: '{train_img_dir}' 和 '{val_img_dir}'")
    print(f"  - **空的** 标签文件 (.txt) 已在: '{train_lbl_dir}' 和 '{val_lbl_dir}' 中创建。")
    print(f"  - **下一步关键操作:** 您现在必须使用标注工具 (例如 LabelImg, Roboflow, CVAT)")
    print(f"    为 train/labels 和 val/labels 目录中的所有 .txt 文件手动添加边界框标注！")
    print(f"  - 标注时，请确保您选择的类别名称/ID 与您的 `data.yaml` 文件中的 `names` 列表完全对应:")

    # 尝试读取 data.yaml 以显示类别名称和ID
    try:
        # 尝试导入 PyYAML
        try:
            import yaml
        except ImportError:
            print("\n    (警告: 需要安装 PyYAML 库 (pip install pyyaml) 才能读取 data.yaml 并显示类别ID)")
            yaml = None # 设置为 None 以跳过后续读取

        if yaml:
            # 假设 data.yaml 与脚本在同一目录或工作区根目录
            # 脚本目录
            script_dir = Path(__file__).parent.resolve()
            # 工作区根目录 (假设脚本在工作区根目录运行)
            workspace_dir = Path.cwd()
            # 输出目录的父目录
            output_parent_dir = Path(output_dir).parent.resolve()

            yaml_path_options = [
                output_parent_dir / 'data.yaml', # data.yaml 在 output_dir 上一级 (最常见)
                workspace_dir / 'data.yaml',     # data.yaml 在运行脚本的当前目录
                script_dir / 'data.yaml',        # data.yaml 和脚本在同一目录
            ]
            yaml_path = None
            for p in yaml_path_options:
                if p.exists() and p.is_file():
                    yaml_path = p
                    break

            if yaml_path:
                print(f"    (读取类别信息自: {yaml_path})")
                try:
                    # 指定 UTF-8 编码读取
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data_yaml = yaml.safe_load(f)
                        if 'names' in data_yaml and isinstance(data_yaml['names'], dict):
                            print("    根据 data.yaml 文件，类别ID应为:")
                            # 按 key (ID) 排序显示
                            for idx in sorted(data_yaml['names'].keys()):
                                 print(f"      {idx}: {data_yaml['names'][idx]}")
                        elif 'names' in data_yaml and isinstance(data_yaml['names'], list):
                             print("    根据 data.yaml 文件，类别ID应为 (从0开始的索引):")
                             for idx, name in enumerate(data_yaml['names']):
                                  print(f"      {idx}: {name}")
                        else:
                             print("    data.yaml 文件中未找到有效的 'names' 列表或字典。")
                except Exception as read_yaml_e:
                     print(f"    读取 data.yaml 时出错: {read_yaml_e}")
            else:
                print(f"    (未能在常见位置找到 data.yaml 文件来显示类别ID)")
                print(f"    请手动参照您的 data.yaml 文件确认类别ID。")

    except Exception as e:
        # 捕获导入 PyYAML 之外的其他潜在错误
        print(f"    尝试读取类别信息时发生意外错误: {e}")
    print("="*72)


if __name__ == "__main__":
    # 使用 argparse 提供命令行接口
    parser = argparse.ArgumentParser(
        description="从分类目录创建 YOLOv8 数据集结构 (包含空的标签文件占位符)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 显示默认值
    )
    parser.add_argument('--source-dir', type=str,
                        default=str(Path("/Users/eric/Desktop/Python/CV/检测 玻璃瓶").resolve()),
                        help="包含类别子目录的源数据路径")
    parser.add_argument('--output-dir', type=str,
                        default=str(Path("/Users/eric/Desktop/Python/CV/检测 玻璃瓶").resolve()),
                        help="输出 YOLO 结构的目标路径 (通常与 source-dir 相同)")
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help="用于训练集的数据比例 (0.0 到 1.0 之间，不含边界)")

    # 检查是否在没有参数的情况下运行（例如，直接双击运行）
    # 如果是，并且 sys.argv 长度为 1，可能需要提示用户或使用默认值
    if len(sys.argv) == 1:
         print("提示：脚本直接运行，将使用默认参数。")
         # 如果希望在无参数时退出或提示，可以在这里添加逻辑
         pass # 继续使用默认值

    args = parser.parse_args()

    # 参数校验
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)

    if not 0 < args.train_ratio < 1:
        print(f"错误：训练比例 (--train-ratio={args.train_ratio}) 必须介于 0 和 1 之间 (例如 0.8 表示 80% 训练集)。")
    elif not source_path.is_dir():
        print(f"错误：源目录 '{args.source_dir}' 不存在或不是一个有效的目录。")
    else:
        # 确保输出目录存在 (如果与源目录不同)
        if source_path.resolve() != output_path.resolve():
             print(f"将在输出目录 '{output_path}' 创建 YOLO 结构。")
             # 在执行前再次确认输出目录存在
             try:
                 output_path.mkdir(parents=True, exist_ok=True)
             except OSError as e:
                  print(f"错误：无法创建输出目录 '{output_path}' - {e}")
                  sys.exit(1) # 退出脚本
        else:
             print(f"将在源目录 '{source_path}' 内部创建 train/ 和 val/ 子目录。")

        # 执行主函数
        create_yolo_structure(str(source_path), str(output_path), args.train_ratio)
 