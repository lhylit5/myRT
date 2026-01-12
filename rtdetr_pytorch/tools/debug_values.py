import os
import sys

# 路径补丁
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import cv2
import numpy as np
from src.core import YAMLConfig
from pycocotools.coco import COCO

# ================= 配置区域 =================
CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
# 使用你做实验的那个 best.pth
CHECKPOINT_PATH =  '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 填入那张"看起来很乱"的图片ID，或者随便填一个
TARGET_IMG_ID = 1584
# ===========================================

def analyze_tensor(name, tensor):
    """打印 Tensor 的统计信息"""
    if tensor is None:
        print(f"[{name}] is None!")
        return

    # 转为 numpy
    data = tensor.float().cpu().detach().numpy()

    print(f"--- 分析: {name} ---")
    print(f"  Shape: {data.shape}")
    print(f"  Min  : {data.min():.6f}")
    print(f"  Max  : {data.max():.6f}")
    print(f"  Mean : {data.mean():.6f}")
    print(f"  Std  : {data.std():.6f}")

    if "density" in name:
        # 对于密度图，看看有多少点是显著激活的 (>0.1)
        active_count = (data > 0.1).sum()
        total_count = data.size
        print(f"  激活像素占比 (>0.1): {active_count / total_count * 100:.2f}%")
    print("-" * 30)

def main():
    print("加载模型...")
    cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)
    model = cfg.model
    if CHECKPOINT_PATH:
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'ema' in state_dict: state_dict = state_dict['ema']['module']
        else: state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    print(f"加载图片 ID: {TARGET_IMG_ID} ...")
    coco = COCO(VAL_ANN_FILE)
    img_info = coco.loadImgs(TARGET_IMG_ID)[0]
    img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])

    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"图片读取失败: {img_path}")
        return

    # 推理
    input_size = 640
    img_resized = cv2.resize(original_img, (input_size, input_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    with torch.no_grad():
        outputs = model(img_tensor)

    # === 开始诊断 ===
    print("\n" + "="*40)
    print("       模型数值诊断报告        ")
    print("="*40 + "\n")

    # 1. 检查密度图 (Pred Density Map)
    # 这是最关键的：它应该大部分是 0，只有少数点 > 1.0
    if 'pred_density_map' in outputs:
        analyze_tensor("Pred Density Map (mask)", outputs['pred_density_map'])
    else:
        print("错误: 模型未输出 pred_density_map")

    # 2. 检查特征图 Before (Base)
    if 'feat_before' in outputs:
        analyze_tensor("Feat Before (Base)", outputs['feat_before'])

    # 3. 检查特征图 After (Enhanced)
    if 'feat_after' in outputs:
        analyze_tensor("Feat After (Ours)", outputs['feat_after'])

    # 4. 计算并检查差值 (Diff)
    if 'feat_after' in outputs and 'feat_before' in outputs:
        # 简单模拟 forward 里的计算: diff = after - before
        # 理论上 diff = feat_before * (alpha * mask)
        # 如果 alpha * mask 很大，diff 就会很大
        diff = outputs['feat_after'] - outputs['feat_before']
        analyze_tensor("Diff (After - Before)", diff)

if __name__ == '__main__':
    main()