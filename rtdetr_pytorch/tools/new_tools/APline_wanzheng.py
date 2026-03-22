import matplotlib.pyplot as plt


# 1. 定义一个函数，用于从指定的日志文件中提取数据
def extract_metrics(log_file_path):
    ap_values = []
    aps_values = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取整体 AP
                if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
                    val = float(line.split('=')[-1].strip())
                    ap_values.append(val)
                # 提取小目标 APsmall
                elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]' in line:
                    val = float(line.split('=')[-1].strip())
                    aps_values.append(val)
    except FileNotFoundError:
        print(f"警告：找不到文件 {log_file_path}，请检查路径。")

    return ap_values, aps_values


# 2. 准备文件路径（请替换为您实际的文件名）
log_file_a = 'E:\experiment\Base_72epo\\train.log'
log_file_b = 'E:\experiment\8card\\traino2m_nta_ndk.log'

# 3. 提取两个模型的数据
ap_a, aps_a = extract_metrics(log_file_a)
ap_b, aps_b = extract_metrics(log_file_b)

# 构建横坐标（以数据较长的那个为准，或者假设两个模型跑了相同的 Epoch）
max_epochs = max(len(ap_a), len(ap_b))
epochs_a = range(1, len(ap_a) + 1)
epochs_b = range(1, len(ap_b) + 1)

# 4. 开始绘图
plt.figure(figsize=(12, 7))

# --- 绘制模型 A 的曲线 (蓝色系) ---
# AP 使用实线和圆圈，APsmall 使用虚线和圆圈
if ap_a:
    plt.plot(epochs_a, ap_a, label='Model A - AP (All)', color='blue', linestyle='-', marker='o', markersize=4)
if aps_a:
    plt.plot(epochs_a, aps_a, label='Model A - AP_small', color='blue', linestyle='--', marker='o', markersize=4)

# --- 绘制模型 B 的曲线 (红色系) ---
# AP 使用实线和方块，APsmall 使用虚线和方块
if ap_b:
    plt.plot(epochs_b, ap_b, label='Model B - AP (All)', color='red', linestyle='-', marker='s', markersize=4)
if aps_b:
    plt.plot(epochs_b, aps_b, label='Model B - AP_small', color='red', linestyle='--', marker='s', markersize=4)

# 5. 设置图表样式
plt.title('Comparison of AP and AP_small: Model A vs Model B', fontsize=15)
plt.xlabel('Epoch (Evaluation Step)', fontsize=12)
plt.ylabel('Average Precision', fontsize=12)

# 添加网格
plt.grid(True, linestyle=':', alpha=0.7)

# 调整图例位置，避免遮挡曲线 (bbox_to_anchor可以将图例放在图表外侧或指定角落)
plt.legend(fontsize=11, loc='lower right')

# 自动调整布局并保存
plt.tight_layout()
plt.savefig('model_comparison_curve.png', dpi=300)
plt.show()