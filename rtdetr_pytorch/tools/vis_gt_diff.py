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
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou

# ================= 配置区域 =================
# 1. Baseline
CONFIG_BASE = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
CKPT_BASE = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

# 2. Ours (创新点)
CONFIG_OURS = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CKPT_OURS = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

# 3. 路径
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 保存路径
SAVE_DIR_SUCCESS = "vis_gt_success"  # Base漏，Ours中
SAVE_DIR_FAILURE = "vis_gt_failure"  # Base中，Ours漏

# 判定阈值
IOU_THRESH = 0.5  # IoU > 0.5 算匹配上 GT
SCORE_THRESH = 0.3  # 置信度 > 0.3 算有效检测


# ===========================================

def load_model(config_path, ckpt_path):
    print(f"Loading: {config_path}")
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    model = cfg.model
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'ema' in state_dict:
            state_dict = state_dict['ema']['module']
        else:
            state_dict = state_dict['model']
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def check_detection(pred_boxes, pred_scores, gt_box_xyxy, iou_thresh, score_thresh):
    """
    检查模型是否检出了某个 GT
    返回: (是否检出, 最佳匹配框, 最佳匹配分数)
    """
    if len(pred_boxes) == 0:
        return False, None, 0.0

    # 筛选低分框
    valid_mask = pred_scores > score_thresh
    valid_boxes = pred_boxes[valid_mask]
    valid_scores = pred_scores[valid_mask]

    if len(valid_boxes) == 0:
        return False, None, 0.0

    # 计算 IoU
    valid_boxes_xyxy = box_cxcywh_to_xyxy(valid_boxes)
    ious = box_iou(valid_boxes_xyxy, gt_box_xyxy)[0].squeeze(1)  # [N]

    max_iou, max_idx = ious.max(dim=0)

    if max_iou > iou_thresh:
        return True, valid_boxes[max_idx], valid_scores[max_idx].item()
    else:
        return False, None, 0.0


def main():
    os.makedirs(SAVE_DIR_SUCCESS, exist_ok=True)
    os.makedirs(SAVE_DIR_FAILURE, exist_ok=True)

    model_base = load_model(CONFIG_BASE, CKPT_BASE)
    model_ours = load_model(CONFIG_OURS, CKPT_OURS)

    coco = COCO(VAL_ANN_FILE)

    # === 新增：获取类别名称映射 ===
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

    print("正在筛选包含小目标的图片...")
    img_ids_with_small = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann['area'] < 32 * 32 and ann['iscrowd'] == 0:
            img_ids_with_small.add(ann['image_id'])
    img_ids = list(img_ids_with_small)
    print(f"待扫描图片总数: {len(img_ids)}")

    cnt_success = 0
    cnt_failure = 0

    for i, img_id in enumerate(img_ids):
        if i % 50 == 0:
            print(f"[{i}/{len(img_ids)}] 逆袭(Ours Win): {cnt_success} | 翻车(Base Win): {cnt_failure}")

        # 1. 获取该图所有 GT
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 筛选出小目标 GT
        small_gts = [ann for ann in anns if ann['area'] < 32 * 32 and ann['iscrowd'] == 0]
        if not small_gts: continue

        # 2. 读取图片 & 推理
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path): continue

        orig_img = cv2.imread(img_path)
        h, w = orig_img.shape[:2]
        input_size = 640
        img_resized = cv2.resize(orig_img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            out_base = model_base(img_tensor)
            out_ours = model_ours(img_tensor)

        # 3. 准备模型预测数据
        base_scores = out_base['pred_logits'].sigmoid().max(-1).values[0]
        base_boxes = out_base['pred_boxes'][0]

        ours_scores = out_ours['pred_logits'].sigmoid().max(-1).values[0]
        ours_boxes = out_ours['pred_boxes'][0]

        # 4. 遍历每一个小目标 GT，进行裁判
        for gt_idx, gt in enumerate(small_gts):
            gx, gy, gw, gh = gt['bbox']
            gt_box_tensor = torch.tensor([gx / w, gy / h, (gx + gw) / w, (gy + gh) / h], device='cuda').unsqueeze(0)

            # 获取类别名
            cat_name = cat_id_to_name.get(gt['category_id'], 'unknown')

            # 裁判：Base 检出了吗？
            base_hit, base_box, base_score = check_detection(base_boxes, base_scores, gt_box_tensor, IOU_THRESH,
                                                             SCORE_THRESH)

            # 裁判：Ours 检出了吗？
            ours_hit, ours_box, ours_score = check_detection(ours_boxes, ours_scores, gt_box_tensor, IOU_THRESH,
                                                             SCORE_THRESH)

            # === 判定逻辑 ===
            case_type = None
            if ours_hit and not base_hit:
                case_type = "success"
                cnt_success += 1
            elif base_hit and not ours_hit:
                case_type = "failure"
                cnt_failure += 1

            # === 可视化保存 ===
            if case_type:
                # 准备底图
                vis_density = np.zeros((h, w, 3), dtype=np.uint8)
                if 'pred_density_map' in out_ours:
                    d = out_ours['pred_density_map'].squeeze().cpu().numpy()
                    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
                    d_uint8 = (d_norm * 255).astype(np.uint8)
                    vis_density = cv2.applyColorMap(cv2.resize(d_uint8, (w, h)), cv2.COLORMAP_JET)

                img_vis_base = orig_img.copy()
                img_vis_ours = cv2.addWeighted(orig_img, 0.6, vis_density, 0.4, 0)

                # 画 GT 框 (白色)
                ix1, iy1, iw, ih = map(int, gt['bbox'])
                cv2.rectangle(img_vis_base, (ix1, iy1), (ix1 + iw, iy1 + ih), (255, 255, 255), 2)
                cv2.rectangle(img_vis_ours, (ix1, iy1), (ix1 + iw, iy1 + ih), (255, 255, 255), 2)

                # === 新增：画 GT 类别名称 ===
                # 放在 GT 框的左上角上方，避免遮挡
                label_text = f"GT: {cat_name}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # 简单的防出界处理
                text_x = max(0, ix1)
                text_y = max(20, iy1 - 5)

                # 画在 Base 图上
                cv2.putText(img_vis_base, label_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # 画在 Ours 图上
                cv2.putText(img_vis_ours, label_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                def draw_pred(img, box, score, color, label):
                    if box is None:
                        # 如果没检出，就在 GT 框旁边写 Miss
                        cv2.putText(img, f"{label}: Miss", (ix1, iy1 + ih + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        return
                    cx, cy, bw, bh = box.cpu().numpy()
                    x1 = int((cx - bw / 2) * w);
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w);
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    # 预测框的分数写在框下面，防止和 GT 文字打架
                    cv2.putText(img, f"{label}: {score:.2f}", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 左图 Base
                draw_pred(img_vis_base, base_box, base_score, (0, 255, 0) if base_hit else (0, 0, 255), "Base")

                # 右图 Ours
                draw_pred(img_vis_ours, ours_box, ours_score, (0, 255, 0) if ours_hit else (0, 0, 255), "Ours")

                combined = np.hstack([img_vis_base, img_vis_ours])
                folder = SAVE_DIR_SUCCESS if case_type == "success" else SAVE_DIR_FAILURE
                # 文件名加上类别，方便查找
                cv2.imwrite(os.path.join(folder, f"{case_type}_{img_id}_{gt_idx}_{cat_name}.png"), combined)

    print("=" * 40)
    print(f"统计完成！")
    print(f"逆袭样本 (Base漏 Ours中): {cnt_success}")
    print(f"翻车样本 (Base中 Ours漏): {cnt_failure}")
    print("=" * 40)


if __name__ == '__main__':
    main()