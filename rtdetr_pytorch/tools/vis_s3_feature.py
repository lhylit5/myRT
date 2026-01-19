import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F

# --- 路径补丁 (保持和你原有代码一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig
from pycocotools.coco import COCO

# ================= 配置区域 =================
# 1. 模型配置
CONFIG_BASE = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
CKPT_BASE = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

CONFIG_OURS = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CKPT_OURS = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

# 2. 数据集路径
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 3. 输出路径
SAVE_DIR = "output/vis_feature_s3_comparison"

# 4. 筛选条件
VIS_COUNT = 50  # 总共可视化多少张图
ONLY_SMALL_OBJECTS = True  # 是否只看包含小目标的图
# ===========================================

# 全局字典用于存储 Hook 截获的特征
feature_maps = {}


def get_encoder_features_hook(name):
    """
    Hook函数：用于截取 HybridEncoder 的输出
    HybridEncoder forward 返回: (outs, pred_density_map, feat_s3_before, feat_s3_after)
    我们需要的是 feat_s3_after (index=3)
    """

    def hook(model, input, output):
        # output 是一个 tuple，根据你的 hybrid_encoder.py 代码
        # output[0]: FPN输出列表
        # output[1]: 密度图
        # output[2]: 增强前的 S3
        # output[3]: 增强后的 S3 (我们需要的)
        if isinstance(output, tuple) and len(output) >= 4:
            feature_maps[name] = output[3]
        else:
            # 兼容性处理：如果 Base 模型用的旧代码可能只返回 outs
            # 但如果你是在同一套代码下运行，只是 config 不同，应该也会返回 tuple
            feature_maps[name] = output[0] if isinstance(output, tuple) else output

    return hook


def load_model_and_register_hook(config_path, ckpt_path, name):
    print(f"Loading {name}: {config_path}")
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    model = cfg.model

    # 加载权重
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'ema' in state_dict:
            state_dict = state_dict['ema']['module']
        else:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)

    model.cuda().eval()

    # 核心：注册 Hook 到 encoder
    # 注意：确保 model.encoder 是存在的。RT-DETR 结构中通常是 model.encoder
    model.encoder.register_forward_hook(get_encoder_features_hook(name))

    return model


def generate_heatmap(feature_tensor, original_img_shape):
    """
    将特征图转换为热力图并叠加在原图大小上
    feature_tensor: [1, C, H_feat, W_feat]
    original_img_shape: (H, W)
    """
    # 1. 在通道维度求平均 (或者取 max)
    # heatmap = torch.max(feature_tensor, dim=1)[0] # Max 响应
    heatmap = torch.mean(feature_tensor, dim=1)  # Mean 响应

    heatmap = heatmap.squeeze().cpu().detach().numpy()  # [H_feat, W_feat]

    # 2. 归一化到 0-255
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)

    # 3. 伪彩色映射
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 4. Resize 到原图大小
    h, w = original_img_shape
    heatmap_resized = cv2.resize(heatmap_color, (w, h))

    return heatmap_resized


def draw_gt_boxes(img, anns, color=(255, 255, 255)):
    """在图上画出 GT 框，用于对照特征激活位置"""
    img_copy = img.copy()
    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    return img_copy


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载模型并注册 Hook
    model_base = load_model_and_register_hook(CONFIG_BASE, CKPT_BASE, 'base')
    model_ours = load_model_and_register_hook(CONFIG_OURS, CKPT_OURS, 'ours')

    # 2. 准备 COCO 数据
    coco = COCO(VAL_ANN_FILE)
    img_ids = coco.getImgIds()

    # 筛选包含小目标的图片
    if ONLY_SMALL_OBJECTS:
        print("筛选包含小目标的图片...")
        small_img_ids = set()
        for ann in coco.loadAnns(coco.getAnnIds()):
            if ann['area'] < 32 * 32 and ann['iscrowd'] == 0:
                small_img_ids.add(ann['image_id'])
        img_ids = list(small_img_ids)

    print(f"待处理图片数量: {len(img_ids)}")

    count = 0
    for img_id in img_ids:
        if count >= VIS_COUNT:
            break

        # 获取标注信息
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        # 只保留小目标用于绘制（或者全部绘制，看你需求）
        small_anns = [ann for ann in anns if ann['area'] < 32 * 32]

        # 如果强调小目标，仅当图中有小目标时才画
        if ONLY_SMALL_OBJECTS and len(small_anns) == 0:
            continue

        # 加载图片
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path): continue

        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]

        # 预处理
        input_size = 640
        img_resized = cv2.resize(orig_img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # 推理 (Hook 会自动捕获特征)
        with torch.no_grad():
            model_base(img_tensor)
            feat_base = feature_maps.get('base')

            model_ours(img_tensor)
            feat_ours = feature_maps.get('ours')

        if feat_base is None or feat_ours is None:
            print(f"Warning: Failed to capture features for image {img_id}")
            continue

        # 生成热力图
        heatmap_base = generate_heatmap(feat_base, (h, w))
        heatmap_ours = generate_heatmap(feat_ours, (h, w))

        # 叠加混合
        # alpha=0.6 原图, beta=0.4 热力图
        vis_base = cv2.addWeighted(orig_img, 0.6, heatmap_base, 0.4, 0)
        vis_ours = cv2.addWeighted(orig_img, 0.6, heatmap_ours, 0.4, 0)

        # 绘制 GT 框 (白色)，方便观察激活是否在物体上
        vis_base = draw_gt_boxes(vis_base, small_anns)
        vis_ours = draw_gt_boxes(vis_ours, small_anns)

        # 添加文字标签
        cv2.putText(vis_base, "Base S3 Feature", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_ours, "Ours S3 Enhanced", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 拼图: [原图(带GT)] [Base] [Ours]
        img_gt_only = draw_gt_boxes(orig_img.copy(), small_anns, color=(0, 255, 0))  # 原图用绿色框
        cv2.putText(img_gt_only, "Original + GT(Small)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack([img_gt_only, vis_base, vis_ours])

        save_path = os.path.join(SAVE_DIR, f"s3_feat_{img_id}.png")
        cv2.imwrite(save_path, combined)

        print(f"Saved: {save_path}")
        count += 1

    print("Visualization Done!")


if __name__ == '__main__':
    main()