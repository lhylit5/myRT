import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
# 可视化密度分数与小目标分数相关性
# === 路径补丁 (根据你的项目结构调整) ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig

# ================= 配置区域 =================
# 1. 配置文件路径
CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
# 2. 模型权重路径 (修改为你最新的权重)
CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/query_select/checkpoint0017.pth'
# 3. COCO 路径
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')
# 4. 输出路径
SAVE_DIR = "output/density_analysis"


# ===========================================

def load_model(config_path, checkpoint_path, device='cuda'):
    print(f"正在加载模型: {config_path}")
    print(f"权重路径: {checkpoint_path}")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = cfg.model
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in state_dict:
            state_dict = state_dict['ema']['module']
        else:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def analyze_dataset(model, coco, device='cuda', max_images=None):
    """
    遍历数据集，收集 (GT数量, 预测密度Sum) 对
    """
    img_ids = coco.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]  # 调试用，只跑前几张

    data_points = []  # list of [gt_count, pred_sum]

    print(f"开始分析 {len(img_ids)} 张图片...")

    for img_id in tqdm(img_ids):
        # 1. 加载图片
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # 2. 预处理 (简单的 Resize 到 640，与训练保持一致)
        # 注意：如果你的 DataLoaders 有特殊的归一化/LetterBox，最好在这里对齐
        input_size = 640
        img_resized = cv2.resize(original_img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # 3. 推理
        with torch.no_grad():
            outputs = model(img_tensor)

        if 'pred_density_map' not in outputs:
            continue

        # 4. 获取预测的密度总质量 (Density Mass)
        # 【关键】因为输出没有 Sigmoid，这里必须先 Sigmoid 再求和
        pred_logits = outputs['pred_density_map']  # [B, 1, H, W]
        pred_prob = torch.sigmoid(pred_logits)
        pred_mass = pred_prob.sum().item()

        # 5. 获取真实的 GT 小目标数量
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        gt_small_count = 0
        gt_total_count = 0

        for ann in anns:
            if ann['iscrowd']: continue
            gt_total_count += 1
            # 统计小目标 (area < 32^2)
            if ann['area'] < 32 * 32:
                gt_small_count += 1

        # 这里你可以选择统计 "小目标数量" 还是 "总目标数量"
        # 既然你的密度图是专门针对小目标的，建议用 gt_small_count
        data_points.append([gt_small_count, pred_mass])

    return np.array(data_points)


def plot_scatter(data, save_dir):
    """
    绘制散点图
    data: shape [N, 2], column 0 is GT, column 1 is Pred
    """
    x = data[:, 0]  # GT Count
    y = data[:, 1]  # Pred Mass

    plt.figure(figsize=(10, 8))

    # 1. 绘制散点
    plt.scatter(x, y, alpha=0.5, s=15, c='dodgerblue', label='Image Samples', edgecolors='white', linewidth=0.5)

    # 2. 绘制理想对角线 (参考线)
    max_val = max(x.max(), y.max())
    plt.plot([0, max_val], [0, max_val], 'k:', alpha=0.3, label='Ideal 1:1')

    # 3. 线性回归拟合 (查看整体趋势)
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        fit_y = slope * x + intercept
        plt.plot(x, fit_y, 'r--', linewidth=2, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        # 计算相关系数
        correlation = np.corrcoef(x, y)[0, 1]
        plt.title(f"Density Map Analysis\nCorrelation: {correlation:.4f}", fontsize=14)
    else:
        plt.title("Density Map Analysis", fontsize=14)

    plt.xlabel("GT Small Object Count (Area < 1024)", fontsize=12)
    plt.ylabel("Predicted Density Sum (Sigmoid Mass)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "scatter_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"散点图已保存: {save_path}")

    # 4. (可选) 绘制分档直方图，帮助你定阈值
    plt.figure(figsize=(12, 6))
    plt.hist(y, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Predicted Density Mass")
    plt.xlabel("Predicted Mass")
    plt.ylabel("Frequency")
    save_path_hist = os.path.join(save_dir, "mass_distribution.png")
    plt.savefig(save_path_hist)
    print(f"分布直方图已保存: {save_path_hist}")


def main():
    # 0. 准备输出目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 2. 加载数据
    print(f"正在加载 COCO 标注: {VAL_ANN_FILE} ...")
    coco = COCO(VAL_ANN_FILE)

    # 3. 运行分析
    # 设置 max_images 可以先跑 100 张看看效果，设为 None 则跑全集
    print("开始推理分析...")
    # data_points = analyze_dataset(model, coco, device, max_images=100)
    data_points = analyze_dataset(model, coco, device, max_images=None)  # 跑全量建议用这个

    # 4. 保存原始数据 (方便后续不用推理直接画图)
    npy_path = os.path.join(SAVE_DIR, "density_data.npy")
    np.save(npy_path, data_points)
    print(f"原始数据已保存: {npy_path}")

    # 5. 画图
    plot_scatter(data_points, SAVE_DIR)

    print("\n=== 分析完成 ===")
    print("请查看 output/density_analysis 下的图表。")
    print("根据散点图的斜率和聚集区域，你可以确定动态选择 Query 的阈值。")


if __name__ == '__main__':
    main()