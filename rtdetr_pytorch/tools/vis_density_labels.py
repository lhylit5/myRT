# 可视化两种策略标签
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# 路径补丁，确保能导入 src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入你写的两种 GT 生成方法
from src.zoo.rtdetr.rtdetr_criterion import generate_density_map_gt, generate_targets_and_weights_adaptive

# ================= 配置区域 =================
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 分别设置保存路径
SAVE_DIR_ORIGINAL = "vis_density_labels/original"  # 存带框的原图
SAVE_DIR_GAUSSIAN = "vis_density_labels/gaussian"  # 存高斯策略图
SAVE_DIR_ADAPTIVE = "vis_density_labels/adaptive"  # 存自适应策略图

INPUT_SIZE = 640
STRIDE = 8  # S3 特征图的下采样倍率 (通常为 8)
FEAT_SIZE = INPUT_SIZE // STRIDE  # 80x80


# ===========================================

def get_normalized_boxes(anns, img_w, img_h):
    """将 COCO [x, y, w, h] 转换为 [cx, cy, w, h] 归一化格式"""
    boxes = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        # 转换为中心点坐标
        cx = x + w / 2.0
        cy = y + h / 2.0
        # 归一化
        boxes.append([cx / img_w, cy / img_h, w / img_w, h / img_h])
    return torch.tensor(boxes, dtype=torch.float32)


def tensor_to_heatmap(tensor_map, target_size=(INPUT_SIZE, INPUT_SIZE)):
    """将 [H, W] 的 Tensor 转换为 Jet 伪彩色热力图"""
    array_map = tensor_map.squeeze().cpu().numpy()
    # 归一化到 0-255 之间
    if array_map.max() > 0:
        array_map = array_map / array_map.max()
    array_map = (array_map * 255).astype(np.uint8)

    # 调整回原图大小
    array_map_resized = cv2.resize(array_map, target_size, interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(array_map_resized, cv2.COLORMAP_JET)
    return heatmap


def main():
    # 创建三个目录
    os.makedirs(SAVE_DIR_ORIGINAL, exist_ok=True)
    os.makedirs(SAVE_DIR_GAUSSIAN, exist_ok=True)
    os.makedirs(SAVE_DIR_ADAPTIVE, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coco = COCO(VAL_ANN_FILE)

    # 找几张包含小目标的图片用于论文展示
    img_ids_with_small = []
    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann['area'] < 32 * 32 and ann['iscrowd'] == 0:
            img_ids_with_small.append(ann['image_id'])

    # 取前几张图生成示例
    sample_img_ids = list(set(img_ids_with_small))[:50]

    for img_id in sample_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        orig_img = cv2.imread(img_path)
        img_h, img_w = orig_img.shape[:2]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # 构造 targets 格式
        boxes_tensor = get_normalized_boxes(anns, img_w, img_h)
        targets = [{'boxes': boxes_tensor.to(device)}]
        feat_shape = (1, FEAT_SIZE, FEAT_SIZE)  # [B, H, W]

        # ---------------------------------------------------------
        # 1. 生成方法 1: 标准高斯分布 (generate_density_map_gt)
        # ---------------------------------------------------------
        density_map_gaussian = generate_density_map_gt(targets, feat_shape, device)
        heatmap_gaussian = tensor_to_heatmap(density_map_gaussian[0])

        # ---------------------------------------------------------
        # 2. 生成方法 2: 自适应 Box-Aware (generate_targets_and_weights_adaptive)
        # ---------------------------------------------------------
        density_map_adaptive, weight_map = generate_targets_and_weights_adaptive(targets, (1, 1, FEAT_SIZE, FEAT_SIZE),
                                                                                 device)
        heatmap_adaptive = tensor_to_heatmap(density_map_adaptive[0])

        # ---------------------------------------------------------
        # 3. 可视化与保存 (分离保存)
        # ---------------------------------------------------------
        img_resized = cv2.resize(orig_img, (INPUT_SIZE, INPUT_SIZE))

        # 画真实框 (为了在对比图里看清楚目标位置)
        for box in boxes_tensor:
            cx, cy, w, h = box.numpy()
            x1 = int((cx - w / 2) * INPUT_SIZE)
            y1 = int((cy - h / 2) * INPUT_SIZE)
            x2 = int((cx + w / 2) * INPUT_SIZE)
            y2 = int((cy + h / 2) * INPUT_SIZE)
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 融合原图和热力图
        blend_gaussian = cv2.addWeighted(img_resized, 0.4, heatmap_gaussian, 0.6, 0)
        blend_adaptive = cv2.addWeighted(img_resized, 0.4, heatmap_adaptive, 0.6, 0)

        # 分别保存到各自的文件夹下
        save_path_orig = os.path.join(SAVE_DIR_ORIGINAL, f"{img_id}.png")
        save_path_gaussian = os.path.join(SAVE_DIR_GAUSSIAN, f"{img_id}.png")
        save_path_adaptive = os.path.join(SAVE_DIR_ADAPTIVE, f"{img_id}.png")

        cv2.imwrite(save_path_orig, img_resized)
        cv2.imwrite(save_path_gaussian, blend_gaussian)
        cv2.imwrite(save_path_adaptive, blend_adaptive)

        print(f"Processed image {img_id}: saved to original/, gaussian/, and adaptive/ directories.")


if __name__ == '__main__':
    main()