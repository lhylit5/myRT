import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# === 路径补丁 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig

# # ================= 配置区域 =================
# CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
# CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')
SAVE_DIR = "output/query_score_analysis"


# ===========================================

def load_model(config_path, checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path}...")
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


def analyze_scores(model, coco, device='cuda', max_images=None):
    img_ids = coco.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]

    all_topk_scores = []  # 存储每张图 Top-300 的分数 [N, 300]

    # 用于 hook 提取 Decoder 内部的分数 (如果你想看中间过程)
    # 这里我们简化，直接看模型输出的 logits (这是最终分数)
    # 注意：RT-DETR 输出的是 pred_logits，未经过 Sigmoid

    print(f"Analyzing {len(img_ids)} images...")

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path): continue

        # 预处理
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        # 获取 pred_logits [B, 300, Num_Classes]
        if 'pred_logits' in outputs:
            logits = outputs['pred_logits']
            # Sigmoid 获取置信度
            scores = logits.sigmoid()
            # 取每个 Query 的最大类别分数
            max_scores, _ = scores.max(dim=-1)  # [B, 300]

            # 转 numpy
            scores_np = max_scores.cpu().numpy().flatten()
            all_topk_scores.append(scores_np)

    return np.array(all_topk_scores)


def plot_analysis(scores, save_dir):
    """
    scores: [Num_Images, 300]
    """
    os.makedirs(save_dir, exist_ok=True)

    # Flatten 所有分数
    flat_scores = scores.flatten()

    # 1. 整体分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(flat_scores, bins=100, color='mediumpurple', alpha=0.7, log=True)  # 使用 log 刻度因为低分可能很多
    plt.title("Distribution of Top-300 Query Scores (Log Scale)")
    plt.xlabel("Score (Confidence)")
    plt.ylabel("Frequency (Log)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axvline(x=0.4, color='r', linestyle='--', label='Threshold 0.4')  # 画个参考线
    plt.legend()
    plt.savefig(os.path.join(save_dir, "score_hist_log.png"))
    plt.close()

    # 2. 也是直方图，但是线性刻度，看高分区域
    plt.figure(figsize=(10, 6))
    plt.hist(flat_scores, bins=100, color='teal', alpha=0.7)
    plt.title("Distribution of Top-300 Query Scores (Linear Scale)")
    plt.xlabel("Score (Confidence)")
    plt.ylabel("Frequency")
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "score_hist_linear.png"))
    plt.close()

    # 3. 每张图 Top-10 vs Top-300 的对比
    # 我们想看"头部" Query 的分数有多高
    top10_scores = scores[:, :10].flatten()
    last10_scores = scores[:, -10:].flatten()  # 排名靠后的

    plt.figure(figsize=(8, 6))
    plt.boxplot([top10_scores, last10_scores], labels=['Top-10 Queries', 'Bottom-10 Queries'])
    plt.title("Score Comparison: Best vs Worst in Top-300")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "score_boxplot.png"))
    plt.close()

    # 4. 打印统计信息
    print("\n=== Score Statistics ===")
    print(f"Mean Score: {np.mean(flat_scores):.4f}")
    print(f"Median Score: {np.median(flat_scores):.4f}")
    print(f"Max Score: {np.max(flat_scores):.4f}")
    print(f"Min Score: {np.min(flat_scores):.4f}")
    print(f"% of Queries > 0.5: {np.mean(flat_scores > 0.5) * 100:.2f}%")
    print(f"% of Queries > 0.1: {np.mean(flat_scores > 0.1) * 100:.2f}%")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    print("Loading COCO...")
    coco = COCO(VAL_ANN_FILE)

    # 加载模型
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 分析
    scores = analyze_scores(model, coco, device, max_images=200)  # 先跑200张看看

    # 绘图
    plot_analysis(scores, SAVE_DIR)
    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == '__main__':
    main()