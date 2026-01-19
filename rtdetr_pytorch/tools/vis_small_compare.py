import os
import sys
import cv2
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import torch
import numpy as np
from src.core import YAMLConfig
from pycocotools.coco import COCO
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy

# ================= 配置区域 =================
# 1. 模型路径 (请确认路径正确)
CONFIG_BASE = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
CKPT_BASE = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

CONFIG_OURS = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CKPT_OURS = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

# 2. 数据配置
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 3. 可视化参数
SAVE_DIR = "vis_final_v3_clear_labels"
CONF_THRES = 0.3  # 预测置信度阈值
IOU_THRES_SMALL = 32 * 32  # 面积阈值，只看小目标
ZOOM_FACTOR = 4  # 放大倍数
ZOOM_SIZE = 160  # 放大窗口大小
LINE_THICKNESS = 1  # 框线粗细 (设为1更精致)


# ===========================================

def load_model(config_path, ckpt_path):
    print(f"Loading: {config_path} ...")
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    model = cfg.model
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'ema' in state_dict:
            state_dict = state_dict['ema']['module']
        else:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def get_predictions(model, img_tensor, orig_size):
    h, w = orig_size
    with torch.no_grad():
        outputs = model(img_tensor)

    prob = outputs['pred_logits'][0].sigmoid()
    scores, labels = prob.max(-1)
    pred_boxes = outputs['pred_boxes'][0]

    # 1. 置信度筛选
    mask = scores > CONF_THRES
    boxes = pred_boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # 坐标还原
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], device=boxes.device)

    # 2. 只有小目标才返回
    wh = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    small_mask = wh < IOU_THRES_SMALL

    return boxes_xyxy[small_mask].cpu().numpy(), scores[small_mask].cpu().numpy(), labels[small_mask].cpu().numpy()


def draw_box(img, box, color, score=None, text=None, font_scale=0.4, font_thickness=1, text_bg_color=None):
    """
    通用画框函数
    :param font_scale: 字体大小，GT图可以设大一点
    :param text_bg_color: 文字背景色，如果不填则使用框的颜色
    """
    x1, y1, x2, y2 = map(int, box)

    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)

    # 绘制标签
    if text or score:
        label = f"{text}" if text else ""
        if score: label += f" {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # 文字位置：默认在框上方，防止遮挡小物体
        # 如果上方没位置了，就放下方
        text_y = y1 - 5
        if text_y < th: text_y = y2 + th + 5

        # 绘制文字背景条
        bg_c = text_bg_color if text_bg_color else color
        # 稍微画大一点背景
        cv2.rectangle(img, (x1, text_y - th - 2), (x1 + tw + 4, text_y + baseline - 2), bg_c, -1)

        # 绘制文字 (白色)
        cv2.putText(img, label, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness)


def add_zoom(img, target_box, color_border):
    """添加局部放大图"""
    if target_box is None: return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, target_box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 计算裁剪区域
    crop_size = int(ZOOM_SIZE / ZOOM_FACTOR)
    x_min = max(0, cx - crop_size // 2)
    y_min = max(0, cy - crop_size // 2)
    x_max = min(w, x_min + crop_size)
    y_max = min(h, y_min + crop_size)

    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0: return img

    # 放大 (使用 NEAREST 保持像素感，或者 LINEAR 平滑)
    zoom_img = cv2.resize(crop, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_NEAREST)
    # 加边框
    cv2.rectangle(zoom_img, (0, 0), (ZOOM_SIZE - 1, ZOOM_SIZE - 1), color_border, 2)

    # 贴图位置：默认右上角
    pad = 10
    paste_x = w - ZOOM_SIZE - pad
    paste_y = pad if cy > h / 2 else (h - ZOOM_SIZE - pad)

    img[paste_y:paste_y + ZOOM_SIZE, paste_x:paste_x + ZOOM_SIZE] = zoom_img

    # 画连接线
    cv2.line(img, (cx, cy), (paste_x, paste_y + ZOOM_SIZE // 2), color_border, 1)

    return img


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_base = load_model(CONFIG_BASE, CKPT_BASE)
    model_ours = load_model(CONFIG_OURS, CKPT_OURS)

    coco = COCO(VAL_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    cat_map = {c['id']: c['name'] for c in cats}

    # 筛选包含小目标的图片
    img_ids = []
    print("Filtering images...")
    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann['area'] < IOU_THRES_SMALL and ann['iscrowd'] == 0:
            img_ids.append(ann['image_id'])
    img_ids = list(set(img_ids))
    print(f"Total images with small objects: {len(img_ids)}")

    count = 0
    for img_id in img_ids:
        if count >= 30: break

        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(fpath): continue

        orig_img = cv2.imread(fpath)
        h, w = orig_img.shape[:2]

        # 1. 准备 GT 数据 (只保留小目标)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        small_gts = []
        for ann in anns:
            if ann['area'] < IOU_THRES_SMALL:
                small_gts.append({
                    'bbox': [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                             ann['bbox'][1] + ann['bbox'][3]],
                    'name': cat_map.get(ann['category_id'], 'obj')
                })

        if not small_gts: continue

        # 2. 推理
        img_resized = cv2.resize(orig_img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        pred_base, score_base, lbl_base = get_predictions(model_base, img_tensor, (h, w))
        pred_ours, score_ours, lbl_ours = get_predictions(model_ours, img_tensor, (h, w))

        if len(pred_ours) == 0: continue

        # ================== 绘图 ==================
        img_col1_gt = orig_img.copy()
        img_col2_base = orig_img.copy()
        img_col3_ours = orig_img.copy()

        # 颜色定义
        COLOR_GT_WHITE = (255, 255, 255)
        COLOR_GT_GRAY = (180, 180, 180)
        COLOR_BASE = (255, 100, 0)  # 蓝色
        COLOR_OURS = (0, 0, 255)  # 红色

        # --- 1. 左图：GT Only (重点优化标签清晰度) ---
        for gt in small_gts:
            # 这里的 font_scale 设为 0.6 (更大)，thickness 设为 1 或 2
            # text_bg_color 设为深灰色，衬托白色文字
            draw_box(img_col1_gt, gt['bbox'], COLOR_GT_WHITE,
                     text=gt['name'],
                     font_scale=0.6,
                     font_thickness=1,
                     text_bg_color=(50, 50, 50))

            # --- 2. 中图：Base (Pred + GT, 有放大) ---
        for gt in small_gts:
            draw_box(img_col2_base, gt['bbox'], COLOR_GT_GRAY)  # GT不写字，只画框
        for i, box in enumerate(pred_base):
            draw_box(img_col2_base, box, COLOR_BASE, score=score_base[i], text=cat_map.get(lbl_base[i], ''))

        # --- 3. 右图：Ours (Pred + GT, 有放大) ---
        for gt in small_gts:
            draw_box(img_col3_ours, gt['bbox'], COLOR_GT_GRAY)

        target_zoom_box = None
        for i, box in enumerate(pred_ours):
            draw_box(img_col3_ours, box, COLOR_OURS, score=score_ours[i], text=cat_map.get(lbl_ours[i], ''))
            if i == 0: target_zoom_box = box

            # --- 添加放大窗口 (只给 Base 和 Ours) ---
        if target_zoom_box is None and len(small_gts) > 0:
            target_zoom_box = small_gts[0]['bbox']

        if target_zoom_box is not None:
            img_col2_base = add_zoom(img_col2_base, target_zoom_box, COLOR_BASE)
            img_col3_ours = add_zoom(img_col3_ours, target_zoom_box, COLOR_OURS)

        # --- 拼接与保存 ---
        def add_header(img, text, bg_color):
            H, W = img.shape[:2]
            header = np.zeros((50, W, 3), dtype=np.uint8) + np.array(bg_color, dtype=np.uint8)
            cv2.putText(header, text, (int(W / 2) - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return np.vstack([header, img])

        final_gt = add_header(img_col1_gt, "Ground Truth", (0, 150, 0))  # 绿头
        final_base = add_header(img_col2_base, "Baseline", (200, 100, 0))  # 蓝头
        final_ours = add_header(img_col3_ours, "Ours", (0, 0, 200))  # 红头

        sep = np.zeros((final_gt.shape[0], 10, 3), dtype=np.uint8) + 255
        combined = np.hstack([final_gt, sep, final_base, sep, final_ours])

        save_path = os.path.join(SAVE_DIR, f"vis_{img_id}.png")
        cv2.imwrite(save_path, combined)
        print(f"Saved: {save_path}")
        count += 1


if __name__ == '__main__':
    main()