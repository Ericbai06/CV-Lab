

import matplotlib.font_manager as fm
import time

print("正在扫描系统字体，这可能需要一点时间...")
start_time = time.time()

# 获取 Matplotlib 字体管理器中所有已注册的字体条目
# ttflist 包含 FontEntry 对象
font_entries = fm.fontManager.ttflist

# 从字体条目中提取唯一的字体名称
# 使用 set 来自动去重，然后转换为列表并排序
font_names = sorted(list(set(f.name for f in font_entries)))

end_time = time.time()
print(f"扫描完成，耗时 {end_time - start_time:.2f} 秒。")
print(f"Matplotlib 共找到 {len(font_names)} 个唯一字体名称。")

print("\n--- Matplotlib 可识别的字体列表 ---")
# 打印所有找到的字体名称
# for name in font_names:
#     print(name)
# 为了避免输出过长，可以选择只打印一部分或者搜索特定字体

print("\n--- 检查特定中文字体和 Arial ---")
target_fonts = ['PingFang SC', 'SimHei', 'Songti SC', 'STSong', 'Arial Unicode MS', 'Arial']
found_count = 0
for font in target_fonts:
    if font in font_names:
        print(f"[✓] {font}: 已找到")
        found_count += 1
    else:
        print(f"[✗] {font}: 未找到")

if found_count == 0 and 'PingFang SC' not in font_names and 'SimHei' not in font_names:
    print("\n警告：未能找到任何目标中文字体 (PingFang SC, SimHei)。这很可能是导致中文显示问题的原因。")
elif 'PingFang SC' not in font_names and 'SimHei' not in font_names:
     print("\n警告：未能找到主要的 PingFang SC 或 SimHei 字体。请确保至少安装了其中一个。")

print("\n提示：如果列表过长，你可以取消上面打印所有字体名称循环的注释，将输出重定向到文件查看 (例如 python your_script.py > fonts.txt)。")
