import os
import sys
import cv2
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
#base与ours对比（ours检测出，base没检测出）
# 路径补丁
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig
from pycocotools.coco import COCO
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou

# ================= 配置区域 (请根据实际情况修改) =================
# 1. 模型路径
CONFIG_BASE = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
CKPT_BASE = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

CONFIG_OURS = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CKPT_OURS = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

# 2. 数据路径
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 3. 输出设置
SAVE_DIR = "vis_results_comparison"
CONF_THRES = 0.35  # 显示框的置信度阈值
ZOOM_FACTOR = 3  # 局部放大的倍数
ZOOM_SIZE = 160  # 局部放大窗口的像素大小 (正方形)


# ==============================================================

def load_model(config_path, ckpt_path):
    print(f"Loading Model: {config_path}...")
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
    """获取模型预测结果，还原到原图尺度"""
    h, w = orig_size
    with torch.no_grad():
        outputs = model(img_tensor)

    # RT-DETR 输出
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]

    prob = pred_logits.sigmoid()
    scores, labels = prob.max(-1)

    # 筛选
    mask = scores > CONF_THRES
    boxes = pred_boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # 归一化坐标 -> 绝对坐标 (cx,cy,w,h) -> (x1,y1,x2,y2)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], device=boxes.device)

    return boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def draw_fancy_box(img, box, score, label_name, color=(0, 255, 0), line_thickness=2):
    """绘制漂亮的边界框和标签"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

    # 标签背景
    text = f"{label_name} {score:.2f}"
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # 避免文字画出界
    txt_x1 = x1
    txt_y1 = y1 - text_h - 4
    if txt_y1 < 0: txt_y1 = y1 + 4

    cv2.rectangle(img, (txt_x1, txt_y1), (txt_x1 + text_w, txt_y1 + text_h + 4), color, -1)
    cv2.putText(img, text, (txt_x1, txt_y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness)


def add_zoom_in_window(img, target_box, zoom_factor=3, win_size=160, color=(0, 0, 255)):
    """
    在图像角落添加局部放大图（画中画）
    target_box: [x1, y1, x2, y2] 需要放大的目标区域
    """
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = map(int, target_box)

    # 1. 计算裁剪区域中心
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 2. 确定 Crop 范围 (保证长宽比一致)
    half_crop = int(win_size / zoom_factor / 2)
    crop_x1 = max(0, cx - half_crop)
    crop_y1 = max(0, cy - half_crop)
    crop_x2 = min(w_img, cx + half_crop)
    crop_y2 = min(h_img, cy + half_crop)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0: return img

    # 3. 放大
    crop_zoomed = cv2.resize(crop, (win_size, win_size), interpolation=cv2.INTER_LINEAR)

    # 4. 给放大图加个边框
    cv2.rectangle(crop_zoomed, (0, 0), (win_size - 1, win_size - 1), color, 4)
    cv2.putText(crop_zoomed, f"Zoom x{zoom_factor}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. 贴回原图 (贴在离目标远一点的角)
    # 简单的策略：如果目标在左边，贴右边；如果目标在上面，贴下面
    paste_x = w_img - win_size - 10 if cx < w_img / 2 else 10
    paste_y = h_img - win_size - 10 if cy < h_img / 2 else 10

    img[paste_y:paste_y + win_size, paste_x:paste_x + win_size] = crop_zoomed

    # 6. 画一条虚线指向放大区域 (可选，这里用简单的实线)
    cv2.line(img, (int(cx), int(cy)), (int(paste_x + win_size / 2), int(paste_y + win_size / 2)), color, 1)

    return img


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载模型
    model_base = load_model(CONFIG_BASE, CKPT_BASE)
    model_ours = load_model(CONFIG_OURS, CKPT_OURS)

    # 加载 COCO 标签
    coco = COCO(VAL_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    cat_map = {c['id']: c['name'] for c in cats}

    # 筛选包含小目标的图片
    print("Filtering images with small objects...")
    img_ids = []
    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann['area'] < 32 * 32:  # 小目标定义
            img_ids.append(ann['image_id'])
    img_ids = list(set(img_ids))
    print(f"Found {len(img_ids)} images with small objects.")

    count_saved = 0

    # 遍历图片
    for i, img_id in enumerate(img_ids):
        if count_saved >= 30: break  # 只保存前30张满意的

        # 加载图片
        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(fpath): continue

        orig_img = cv2.imread(fpath)
        h, w = orig_img.shape[:2]

        # 预处理
        img_resized = cv2.resize(orig_img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # 推理
        box_base, score_base, lbl_base = get_predictions(model_base, img_tensor, (h, w))
        box_ours, score_ours, lbl_ours = get_predictions(model_ours, img_tensor, (h, w))

        # 获取 GT 用于判断 (可选)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_anns = coco.loadAnns(ann_ids)
        box_gt = []
        for ann in gt_anns:
            x, y, bw, bh = ann['bbox']
            # 只关心小目标 GT
            if ann['area'] < 32 * 32:
                box_gt.append([x, y, x + bw, y + bh])

        # =========================================================
        # 寻找“亮点”逻辑：
        # 我们想找到一个 Ours 检出 (Score高) 但 Base 没检出 (IOU=0) 的框
        # 并且这个框最好是匹配上了 GT 的 (是 True Positive)
        # =========================================================
        best_diff_idx = -1
        max_score_diff = 0

        for idx_o, b_o in enumerate(box_ours):
            # 1. 这个 Ours 框必须是有效的小目标 (面积判断略，因为前面已经筛了)
            # 2. 检查 Base 有没有对应的框
            is_missed_by_base = True
            if len(box_base) > 0:
                ious = box_iou(torch.tensor([b_o]), torch.tensor(box_base))[0]
                if ious.max() > 0.5:  # Base 也检出了
                    is_missed_by_base = False

            # 3. 检查是否匹配 GT (确保不是误检)
            is_true_positive = False
            if len(box_gt) > 0:
                ious_gt = box_iou(torch.tensor([b_o]), torch.tensor(box_gt))[0]
                if ious_gt.max() > 0.3:  # 匹配上了某个小目标 GT
                    is_true_positive = True

            # 策略：只要 Base 没检出，且 Ours 检出了，或者 Ours 分数显著高于 Base
            if is_missed_by_base and is_true_positive:
                best_diff_idx = idx_o
                break  # 找到一个就够了

        # 只有当发现明显的改进时才保存
        if best_diff_idx != -1:
            # 绘图准备
            canvas_base = orig_img.copy()
            canvas_ours = orig_img.copy()

            # 画 GT (灰色虚线效果，用实线代替，颜色淡一点)
            gt_color = (200, 200, 200)
            for b in box_gt:
                cv2.rectangle(canvas_base, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), gt_color, 1)
                cv2.rectangle(canvas_ours, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), gt_color, 1)

            # 画 Base 预测 (蓝色)
            for j, b in enumerate(box_base):
                draw_fancy_box(canvas_base, b, score_base[j], cat_map.get(lbl_base[j], 'obj'), (255, 100, 0))

            # 画 Ours 预测 (红色)
            target_box_for_zoom = None
            for j, b in enumerate(box_ours):
                # 如果是那个关键的差异框，用亮红色，否则用暗红色
                is_highlight = (j == best_diff_idx)
                color = (0, 0, 255) if is_highlight else (0, 0, 180)
                thickness = 3 if is_highlight else 2
                draw_fancy_box(canvas_ours, b, score_ours[j], cat_map.get(lbl_ours[j], 'obj'), color, thickness)

                if is_highlight:
                    target_box_for_zoom = b

            # === 注入灵魂：局部放大 ===
            if target_box_for_zoom is not None:
                canvas_ours = add_zoom_in_window(canvas_ours, target_box_for_zoom,
                                                 zoom_factor=ZOOM_FACTOR, win_size=ZOOM_SIZE, color=(0, 0, 255))
                # 为了对比，也在 Base 图对应位置画个 Zoom，看看它是空的
                canvas_base = add_zoom_in_window(canvas_base, target_box_for_zoom,
                                                 zoom_factor=ZOOM_FACTOR, win_size=ZOOM_SIZE, color=(255, 100, 0))

            # 拼接
            h_c, w_c = canvas_base.shape[:2]
            separator = np.zeros((h_c, 10, 3), dtype=np.uint8) + 255  # 白色分割线
            combined = np.hstack([canvas_base, separator, canvas_ours])

            # 顶部加标题
            header = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, "Baseline (Blue: Pred, Gray: GT)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.putText(header, "Ours (Red: Pred + Zoom-in)", (w_c + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            final_img = np.vstack([header, combined])

            save_path = os.path.join(SAVE_DIR, f"compare_{img_id}_score{score_ours[best_diff_idx]:.2f}.png")
            cv2.imwrite(save_path, final_img)
            print(f"Saved comparison: {save_path}")
            count_saved += 1


if __name__ == '__main__':
    main()