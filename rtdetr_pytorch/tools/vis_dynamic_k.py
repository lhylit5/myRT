import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# 可视化一对多匹配，原版/动态k
# === 路径补丁 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy, generalized_box_iou
import torch.nn as nn

# ================= 配置区域 =================
# 建议用带 SE 增强的权重，因为它预测的框更准，对比效果更明显
CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
CHECKPOINT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')
SAVE_DIR = "output/dynamic_k_visualization"


# ===========================================

class TaskAlignedDynamicKMatcher(nn.Module):
    """
    支持消融实验的多功能 O2M 匹配器
    """

    def __init__(self, alpha=1.0, beta=1.0, topk_candidates=10, max_k=7,
                 use_task_align=False, use_dynamic_k=True, fixed_k=6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk_candidates = topk_candidates
        self.max_k = max_k
        self.use_task_align = use_task_align
        self.use_dynamic_k = use_dynamic_k
        self.fixed_k = fixed_k

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].sigmoid()
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            if len(tgt_ids) == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64, device=out_prob.device),
                                torch.as_tensor([], dtype=torch.int64, device=out_prob.device)))
                continue

            tgt_bbox = targets[b]["boxes"]
            num_gt = len(tgt_ids)

            pred_prob = out_prob[b]
            cls_scores = pred_prob[:, tgt_ids]
            iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))

            # 1. 对齐得分计算
            if self.use_task_align:
                align_metric = (cls_scores ** self.alpha) * ((iou.clamp(min=0) + 1e-6) ** self.beta)
                out_cxcy = out_bbox[b][:, :2]
                tgt_cxcy = tgt_bbox[:, :2]
                dist = torch.cdist(out_cxcy, tgt_cxcy)
                align_metric = align_metric / (dist + 1e-6)
            else:
                alpha_focal, gamma_focal = 0.25, 2.0
                neg_cost_class = (1 - alpha_focal) * (pred_prob ** gamma_focal) * (-(1 - pred_prob + 1e-8).log())
                pos_cost_class = alpha_focal * ((1 - pred_prob) ** gamma_focal) * (-(pred_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))
                cost_matrix = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
                align_metric = -cost_matrix

            # 2. 动态/固定 K 值
            if self.use_dynamic_k:
                topk_iou, _ = torch.topk(iou, min(self.topk_candidates, num_queries), dim=0)
                dynamic_ks = torch.clamp(topk_iou.sum(0).int(), min=1, max=self.max_k)
            else:
                dynamic_ks = torch.full((num_gt,), self.fixed_k, dtype=torch.int32, device=out_prob.device)

            # 3. 分配逻辑
            matching_matrix = torch.zeros_like(align_metric, dtype=torch.bool)
            for gt_idx in range(num_gt):
                k = dynamic_ks[gt_idx].item()
                _, topk_idx = torch.topk(align_metric[:, gt_idx], k, largest=True)
                matching_matrix[topk_idx, gt_idx] = True

            # anchor_matching_gt = matching_matrix.sum(1)
            # if (anchor_matching_gt > 1).sum() > 0:
            #     conflict_indices = torch.where(anchor_matching_gt > 1)[0]
            #     for c_idx in conflict_indices:
            #         matched_gts = torch.where(matching_matrix[c_idx])[0]
            #         best_gt = matched_gts[torch.argmax(align_metric[c_idx, matched_gts])]
            #         matching_matrix[c_idx] = False
            #         matching_matrix[c_idx, best_gt] = True

            src_ind, tgt_ind = torch.where(matching_matrix)
            indices.append((src_ind, tgt_ind))

        return indices


def load_model(config_path, checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path}...")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = cfg.model
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = state_dict.get('ema', {}).get('module', state_dict.get('model', state_dict))
        model.load_state_dict(state_dict, strict=False)  # 加上 strict=False
    model.to(device)
    model.eval()
    return model


def get_targets(coco, img_id, device):
    cat_ids = coco.getCatIds()
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(img_id)[0]
    img_h, img_w = img_info['height'], img_info['width']

    boxes, labels = [], []
    for ann in anns:
        if ann.get('ignore', 0) == 1 or ann['area'] <= 0: continue
        if ann['category_id'] not in cat2label: continue  # 过滤无效类别

        x, y, w, h = ann['bbox']
        # 转为 cxcywh normalized
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        boxes.append([cx, cy, w / img_w, h / img_h])

        labels.append(cat2label[ann['category_id']])

    if len(boxes) == 0:
        return [{'labels': torch.tensor([], dtype=torch.int64, device=device),
                 'boxes': torch.tensor([], dtype=torch.float32, device=device).reshape(0, 4)}]

    return [{'labels': torch.tensor(labels, dtype=torch.int64, device=device),
             'boxes': torch.tensor(boxes, dtype=torch.float32, device=device)}]


def draw_boxes_on_ax(ax, img, gt_boxes, pred_boxes, tgt_inds, title, fixed_k=None):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')
    img_h, img_w = img.shape[:2]

    # 1. 绘制 GT (实线，绿色)
    for i, box in enumerate(gt_boxes):
        cx, cy, w, h = box
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        w_abs, h_abs = w * img_w, h * img_h

        # 为了更清楚，我们可以用不同颜色区分最大框和最小框，默认统一绿色
        rect = patches.Rectangle((x1, y1), w_abs, h_abs, linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        # 统计分配给这个 GT 的 Query 数量
        matched_count = (tgt_inds == i).sum().item()

        # 在 GT 框旁边标出 K 值
        ax.text(x1, y1 - 5, f'K={matched_count}', color='lime', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

    # 2. 绘制匹配的预测框 (虚线，红色/橙色)
    for i, box in enumerate(pred_boxes):
        cx, cy, w, h = box
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        w_abs, h_abs = w * img_w, h * img_h

        # 如果是固定K=6，强行分配的偏离框，画成红色虚线以示“背景噪声”
        rect = patches.Rectangle((x1, y1), w_abs, h_abs, linewidth=1.5, linestyle='--', edgecolor='red',
                                 facecolor='none', alpha=0.9)
        ax.add_patch(rect)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading COCO...")
    coco = COCO(VAL_ANN_FILE)
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # 匹配器 A: 模拟固定 K 值 (固定 K=6)
    matcher_fixed = TaskAlignedDynamicKMatcher(use_task_align=False, use_dynamic_k=False, fixed_k=6).to(device)
    # 匹配器 B: 你的创新点 (动态 K + 任务对齐)
    matcher_dynamic = TaskAlignedDynamicKMatcher(use_task_align=True, use_dynamic_k=True).to(device)

    img_ids = coco.getImgIds()

    # 替换为你想要画的图片的 ID，例如 25424
    # for img_id in [:50]:
    for img_id in tqdm(img_ids[:100]):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None: continue

        img_resized = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        targets = get_targets(coco, img_id, device)
        if len(targets[0]['boxes']) == 0: continue

        with torch.no_grad():
            outputs = model(img_tensor)

            outputs_for_match = {
                'pred_logits': outputs['pred_logits'],
                'pred_boxes': outputs['pred_boxes']
            }

            indices_fixed = matcher_fixed(outputs_for_match, targets)
            indices_dynamic = matcher_dynamic(outputs_for_match, targets)

        # 提取数据准备过滤
        gt_boxes = targets[0]['boxes'].cpu().numpy()
        pred_boxes = outputs_for_match['pred_boxes'][0].cpu().numpy()

        src_ind_f = indices_fixed[0][0].cpu().numpy()
        tgt_ind_f = indices_fixed[0][1].cpu().numpy()
        src_ind_d = indices_dynamic[0][0].cpu().numpy()
        tgt_ind_d = indices_dynamic[0][1].cpu().numpy()

        # ==============================================================
        # 【核心修改区】：查找最大和最小 GT，过滤无用的框，只保留这俩的匹配结果
        # ==============================================================
        if len(gt_boxes) >= 2:
            # 计算面积 (w * h)，注意 boxes 是 cx, cy, w, h
            areas = gt_boxes[:, 2] * gt_boxes[:, 3]
            max_idx = np.argmax(areas)
            min_idx = np.argmin(areas)

            if max_idx != min_idx:
                keep_gt_indices = [max_idx, min_idx]
            else:
                keep_gt_indices = [max_idx]
        else:
            keep_gt_indices = list(range(len(gt_boxes)))

        # 定义一个过滤函数
        def filter_matches(src, tgt, keep_inds):
            mask = np.isin(tgt, keep_inds)
            f_src = src[mask]
            f_tgt = tgt[mask]
            # 重新映射 tgt index 以匹配新的 gt_boxes 数组
            remapped_tgt = np.zeros_like(f_tgt)
            for new_i, old_i in enumerate(keep_inds):
                remapped_tgt[f_tgt == old_i] = new_i
            return f_src, remapped_tgt

        # 应用过滤
        src_ind_f, tgt_ind_f = filter_matches(src_ind_f, tgt_ind_f, keep_gt_indices)
        src_ind_d, tgt_ind_d = filter_matches(src_ind_d, tgt_ind_d, keep_gt_indices)

        # 过滤 GT 框
        gt_boxes = gt_boxes[keep_gt_indices]
        # ==============================================================

        # 根据过滤后的 src 索引获取最终的预测框
        pred_boxes_fixed = pred_boxes[src_ind_f]
        pred_boxes_dynamic = pred_boxes[src_ind_d]

        # ========== 开始绘图 ==========
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Image ID: {img_id} - Fixed K vs Dynamic K Matching (Min & Max Objects)", fontsize=16,
                     fontweight='bold')

        # 左图：Fixed K
        draw_boxes_on_ax(axes[0], img_resized, gt_boxes, pred_boxes_fixed, tgt_ind_f,
                         "(a) Fixed-K Matching (K=6)", fixed_k=6)

        # 右图：Dynamic K
        draw_boxes_on_ax(axes[1], img_resized, gt_boxes, pred_boxes_dynamic, tgt_ind_d,
                         "(b) Task-Aligned Dynamic-K Matching (Ours)")

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"compare_{img_id}_minmax.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nVisualization images saved to {SAVE_DIR}. Look for the 'minmax' suffix file!")


if __name__ == '__main__':
    main()