import os
import sys

# 路径补丁
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import cv2
import numpy as np
import random
from pycocotools.coco import COCO
from src.core import YAMLConfig

# ================= 配置区域 =================
# 1. 配置文件路径
CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
# 2. 模型权重路径
CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'
# CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/query_select/checkpoint0061.pth'
# CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/query_select/checkpoint0061.pth'
# 3. COCO 路径
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')
# 4. 保存路径
SAVE_DIR = "output/vis_paper_results"

# === 新增功能：指定图片 ID ===
# 如果列表不为空，脚本将【只可视化】这些 ID 的图片 (用于论文定向截图)
# 如果列表为空 []，脚本将【自动随机】抽取包含小目标的图片
SPECIFIC_IMG_IDS = [139099,192670]
# 示例：SPECIFIC_IMG_IDS = [397133, 37777, 50012]

# 随机模式下的采样数量
NUM_RANDOM_SAMPLES = 8


# ===========================================

def get_all_small_object_ids(coco):
    """从 COCO 中找出所有包含小目标 (area < 32*32) 的图片 ID"""
    small_img_ids = set()
    # 获取所有标注
    ann_ids = coco.getAnnIds()
    # 这一步可能会慢一点，加载所有标注
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if ann['area'] < 32 * 32 and ann['iscrowd'] == 0:
            small_img_ids.add(ann['image_id'])

    return list(small_img_ids)


def visualize_for_paper(model, img_ids, coco, device='cuda'):
    model.eval()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"准备可视化 {len(img_ids)} 张图片，保存至 {SAVE_DIR} ...")

    for i, img_id in enumerate(img_ids):
        # 1. 检查图片是否存在
        img_info_list = coco.loadImgs(img_id)
        if not img_info_list:
            print(f"Warning: ID {img_id} 在标注文件中未找到信息，跳过。")
            continue
        img_info = img_info_list[0]

        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Warning: 图片文件不存在: {img_path}，跳过。")
            continue

        original_img = cv2.imread(img_path)
        if original_img is None: continue
        h, w = original_img.shape[:2]

        # 2. 推理获取密度图
        input_size = 640
        # 简单缩放 (注意：这里为了对齐特征图，直接resize，如果你的模型推理必须用 letterbox 请相应调整)
        img_resized = cv2.resize(original_img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        if 'pred_density_map' not in outputs:
            print("Error: 模型输出字典中没有 'pred_density_map'。请检查 rtdetr.py 是否修改了 export 逻辑。")
            return

        # 3. 处理密度图
        density = outputs['pred_density_map'].cpu().squeeze().numpy()

        # 归一化 (0-255) 以便可视化
        # 注意：如果使用了面积加权，小目标处的值可能 > 1，这里 min-max 归一化会拉伸对比度
        den_min, den_max = density.min(), density.max()
        if den_max > den_min:
            den_norm = (density - den_min) / (den_max - den_min)
        else:
            den_norm = density  # 全0

        den_uint8 = (den_norm * 255).astype(np.uint8)
        # 上采样回原图尺寸
        heatmap = cv2.applyColorMap(cv2.resize(den_uint8, (w, h)), cv2.COLORMAP_JET)

        # 4. 叠加显示
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # 5. 画 GT 框
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 准备一张纯净的原图用于画框对比
        img_with_box = original_img.copy()

        for ann in anns:
            x, y, bw, bh = map(int, ann['bbox'])
            # 小目标用绿色 (显著)，大目标用白色 (作为背景参考)
            is_small = ann['area'] < 32 * 32
            color = (0, 255, 0) if is_small else (200, 200, 200)
            thickness = 2 if is_small else 1

            # 在叠加图上画框
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness)
            # 在原图上画框
            cv2.rectangle(img_with_box, (x, y), (x + bw, y + bh), color, thickness)

        # 6. 拼接 (左: 原图+GT, 右: 密度图+GT)
        combined = np.hstack([img_with_box, overlay])

        save_path = os.path.join(SAVE_DIR, f"vis_{img_id}.png")
        cv2.imwrite(save_path, combined)
        print(f"[{i + 1}/{len(img_ids)}] 已保存: {save_path}")


def main():
    # 1. 加载模型
    print("正在加载模型权重...")
    cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)
    model = cfg.model
    if CHECKPOINT_PATH:
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'ema' in state_dict:
            state_dict = state_dict['ema']['module']
        else:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    model.cuda()

    # 2. 加载 COCO 标注
    print(f"正在加载 COCO 标注: {VAL_ANN_FILE} ...")
    coco = COCO(VAL_ANN_FILE)

    # 3. 决定要可视化的图片 ID
    if SPECIFIC_IMG_IDS:
        print(f"模式：【指定 ID 可视化】")
        print(f"目标 ID: {SPECIFIC_IMG_IDS}")
        target_ids = SPECIFIC_IMG_IDS
    else:
        print(f"模式：【随机抽取小目标图片】")
        all_small_ids = get_all_small_object_ids(coco)
        print(f"共发现 {len(all_small_ids)} 张包含小目标的图片。")

        # 随机抽取
        count = min(NUM_RANDOM_SAMPLES, len(all_small_ids))
        target_ids = random.sample(all_small_ids, count)
        print(f"已随机抽取 {count} 张。")

    # 4. 执行可视化
    visualize_for_paper(model, target_ids, coco)
    print("所有任务完成！")


if __name__ == '__main__':
    main()