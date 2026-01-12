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
# 务必使用那个 AP_S 最高的模型
CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')
SAVE_DIR = "vis_diff_results/se"

# 填入你觉得有小目标的图片 ID
TARGET_IMG_IDS = [139099,192670,1584,29393]


# ===========================================

def get_heatmap(feat, mode='max'):
    """
    feat: [C, H, W]
    mode: 'max' (推荐) 或 'mean'
    return: [H, W] numpy array (未归一化)
    """
    if mode == 'max':
        # 取通道维度的最大值，最能反映稀疏的小目标特征
        heatmap, _ = torch.max(feat, dim=0)
    else:
        heatmap = torch.mean(feat, dim=0)
    return heatmap.cpu().detach().numpy()


def visualize_diff(model, img_id, coco, device='cuda'):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 准备图片
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
    original_img = cv2.imread(img_path)
    h, w = original_img.shape[:2]

    input_size = 640
    img_resized = cv2.resize(original_img, (input_size, input_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 2. 推理
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    if 'feat_before' not in outputs:
        print("Error: 没找到 feat_before，请确认 rtdetr.py 已修改返回特征图")
        return

    # 3. 获取特征并转为单通道热力图 (使用 raw values)
    # 建议使用 'max' 模式，因为小目标特征往往很尖锐
    raw_before = get_heatmap(outputs['feat_before'][0], mode='max')
    raw_after = get_heatmap(outputs['feat_after'][0], mode='max')

    # 4. 计算差值 (After - Before)
    # 这里的正值表示增强，负值表示抑制
    # 原来是: raw_diff = raw_after - raw_before
    # 改为:
    raw_diff = np.abs(raw_after) - np.abs(raw_before)

    # 5. 归一化策略
    # 策略 A: 统一归一化 (为了对比 Before 和 After 的亮度差异)
    # 找出两张图共同的 min 和 max，这样如果 After 变强了，它会显得更亮
    global_min = min(raw_before.min(), raw_after.min())
    global_max = max(raw_before.max(), raw_after.max())

    def normalize_global(hm):
        norm = (hm - global_min) / (global_max - global_min + 1e-6)
        return (norm * 255).astype(np.uint8)

    vis_before = cv2.applyColorMap(cv2.resize(normalize_global(raw_before), (w, h)), cv2.COLORMAP_JET)
    vis_after = cv2.applyColorMap(cv2.resize(normalize_global(raw_after), (w, h)), cv2.COLORMAP_JET)

    # 策略 B: 差值图归一化 (重点展示变化)
    # 我们希望 0 对应灰色/蓝色，正数对应红色
    # 简单起见，直接 min-max 归一化差值图
    diff_norm = (raw_diff - raw_diff.min()) / (raw_diff.max() - raw_diff.min() + 1e-6)
    vis_diff = cv2.applyColorMap(cv2.resize((diff_norm * 255).astype(np.uint8), (w, h)), cv2.COLORMAP_JET)

    # 6. 画 GT 框
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_gt = original_img.copy()
    for ann in anns:
        x, y, bw, bh = map(int, ann['bbox'])
        color = (0, 255, 0) if ann['area'] < 32 * 32 else (200, 200, 200)
        cv2.rectangle(img_gt, (x, y), (x + bw, y + bh), color, 2)
        # 在差值图上也画框，方便定位
        cv2.rectangle(vis_diff, (x, y), (x + bw, y + bh), (255, 255, 255), 1)

    # 7. 拼接: 原图 | Before | After | Diff
    def add_title(img, text):
        pad_img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(pad_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return pad_img

    final_img = np.hstack([
        add_title(img_gt, "GT"),
        add_title(vis_before, "Before (Base)"),
        add_title(vis_after, "After (Enhance)"),
        add_title(vis_diff, "Diff (After-Before)")
    ])

    save_path = os.path.join(SAVE_DIR, f"diff_{img_id}.png")
    cv2.imwrite(save_path, final_img)
    print(f"[{img_id}] 差值对比图已保存: {save_path}")


def main():
    print("加载模型...")
    cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)
    model = cfg.model
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'ema' in state_dict:
        state_dict = state_dict['ema']['module']
    else:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model.cuda()

    coco = COCO(VAL_ANN_FILE)

    # 如果没有指定 ID，随机找几个小目标图
    ids = TARGET_IMG_IDS
    if not ids:
        import random
        all_ids = coco.getImgIds()
        ids = random.sample(all_ids, 5)

    for img_id in ids:
        visualize_diff(model, img_id, coco)


if __name__ == '__main__':
    main()