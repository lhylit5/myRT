"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from torchvision.ops import sigmoid_focal_loss

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register

from .visualize import visualize_boxes


# ==========================================
# === 创新点三：密度引导的一对多匹配器 ===
# ==========================================



# === 辅助函数：处理多卡归一化 ===
def get_world_size():
    if not dist.is_available(): return 1
    if not dist.is_initialized(): return 1
    return dist.get_world_size()


def reduce_mean(tensor):
    """
    多卡同步：将所有卡上的 tensor 求和，然后除以 GPU 数量。
    用于 Loss 的同步，或者统计数量。
    """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)  # 默认是 Sum 操作
        return tensor / world_size


@register
class TaskAlignedDynamicKMatcher(nn.Module):
    """
    支持消融实验的多功能 O2M 匹配器。
    可以通过开关控制：
    1. use_task_align: True 使用任务对齐分数 (你的创新点) / False 使用 MS-DETR 的传统 Cost 矩阵
    2. use_dynamic_k: True 使用动态 K 值 (你的创新点) / False 使用固定 K 值
    """

    def __init__(self,
                 alpha=1.0, beta=1.0, topk_candidates=10, max_k=7,
                 use_task_align=False,  # <--- 消融实验开关 1
                 use_dynamic_k=True,  # <--- 消融实验开关 2
                 fixed_k=6):  # <--- 如果不用动态 K，固定的 K 值设为多少
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk_candidates = topk_candidates
        self.max_k = max_k

        # 消融实验参数
        self.use_task_align = use_task_align
        self.use_dynamic_k = use_dynamic_k
        self.fixed_k = fixed_k

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].sigmoid()  # [B, num_queries, num_classes]
        out_bbox = outputs["pred_boxes"]  # [B, num_queries, 4]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            if len(tgt_ids) == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64, device=out_prob.device),
                                torch.as_tensor([], dtype=torch.int64, device=out_prob.device)))
                continue

            tgt_bbox = targets[b]["boxes"]
            num_gt = len(tgt_ids)

            # 提取概率与计算 IoU (两种方法都需要用到)
            pred_prob = out_prob[b]
            cls_scores = pred_prob[:, tgt_ids]
            iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))

            # =======================================================
            # === 消融维度 1：任务对齐 (Task Alignment) vs 传统 Cost ===
            # =======================================================
            if self.use_task_align:
                #  任务对齐得分 (越大越好)
                # align_metric = (cls_scores ** self.alpha) * ((iou.clamp(min=0) + 1e-6) ** self.beta)
                # # 引入空间先验 (防冷启动)
                # out_cxcy = out_bbox[b][:, :2]
                # tgt_cxcy = tgt_bbox[:, :2]
                # dist = torch.cdist(out_cxcy, tgt_cxcy)
                # align_metric = align_metric / (dist + 1e-6)

                # 1. 提取预测框和真实框的中心点
                out_cxcy = out_bbox[b][:, :2]
                tgt_cxcy = tgt_bbox[:, :2]
                # 2. 计算中心点欧氏距离矩阵 [num_queries, num_gt]
                dist = torch.cdist(out_cxcy, tgt_cxcy)
                # 3. 计算真实目标框(GT)的归一化面积 [num_gt]
                gt_areas = tgt_bbox[:, 2] * tgt_bbox[:, 3]
                # 4. 构建尺度自适应的高斯核 (Scale-Adaptive Gaussian Kernel)
                # 目标越小，高斯核相对其面积就越宽容 (sigma 决定了高斯山的胖瘦)
                # clamp(min=1e-4) 是为了防止极小目标导致除以零
                sigma = torch.clamp(torch.sqrt(gt_areas) * 0.5, min=1e-4)
                # 5. 计算高斯空间相似度 (Gaussian Similarity)
                # 结果在 0 到 1 之间。距离越近越接近1，距离越远平滑衰减到0
                gaussian_sim = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
                # 6. 计算 Scale-Adaptive Gaussian Compensated IoU  (核心创新：空间相似度补偿)
                # 初始化 sgc_iou 为真实的 iou
                sgc_iou = iou.clone().clamp(min=0)
                # 筛选出微小目标
                threshold = 0.01
                is_small = gt_areas < threshold
                # 对微小目标应用 "sgc_iou" 融合
                # 即使真实 IoU 为 0，只要空间距离近(gaussian_sim>0)，依然能获得基础得分
                # 这里的 0.5 是权重，表示真实 IoU 和空间相似度各占一半
                if is_small.any():
                    # 计算动态权重: 目标越小，高斯权重越大；目标接近 threshold，高斯权重接近 0
                    # 比如：面积为 0，权重为 0.5；面积为 0.04，权重为 0
                    gauss_weight = 0.5 * (1.0 - gt_areas[is_small] / threshold)
                    iou_weight = 1.0 - gauss_weight
                    # 动态融合
                    sgc_iou[:, is_small] = iou_weight * iou[:, is_small].clamp(min=0) + gauss_weight * gaussian_sim[:, is_small]
                # 7. 计算最终的任务对齐得分 (Task Alignment Metric)
                # 完全摒弃分母的刚性距离惩罚，采用软化后的 sgc_iou
                align_metric = (cls_scores ** self.alpha) * ((sgc_iou + 1e-6) ** self.beta)
            else:
                # [Baseline] 类似 MS-DETR，计算传统的 Matching Cost (包含类别 Focal Cost, L1, GIoU)
                # 1. Focal Loss Cost
                alpha_focal = 0.25
                gamma_focal = 2.0
                neg_cost_class = (1 - alpha_focal) * (pred_prob ** gamma_focal) * (-(1 - pred_prob + 1e-8).log())
                pos_cost_class = alpha_focal * ((1 - pred_prob) ** gamma_focal) * (-(pred_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

                # 2. BBox L1 Cost
                cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

                # 3. GIoU Cost
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))

                # 融合 Cost (依照 DETR 默认权重: cls=2.0, bbox=5.0, giou=2.0)
                cost_matrix = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou

                # 因为后面的代码都是“越大越好 (largest=True)”，所以这里对 Cost 取负数
                align_metric = -cost_matrix

            # =======================================================
            # === 消融维度 2：动态 K (Dynamic-K) vs 固定 K (Fixed-K) ===
            # =======================================================
            if self.use_dynamic_k:
                # [你的创新点] 动态 K 值计算
                topk_iou, _ = torch.topk(iou, min(self.topk_candidates, num_queries), dim=0)
                dynamic_ks = torch.clamp(topk_iou.sum(0).int(), min=1, max=self.max_k)
            else:
                # [Baseline] 固定 K 值分配
                dynamic_ks = torch.full((num_gt,), self.fixed_k, dtype=torch.int32, device=out_prob.device)

            # =======================================================
            # === 后续分配逻辑 (保持一致，方便公平对比) ===
            # =======================================================
            matching_matrix = torch.zeros_like(align_metric, dtype=torch.bool)

            for gt_idx in range(num_gt):
                k = dynamic_ks[gt_idx].item()
                _, topk_idx = torch.topk(align_metric[:, gt_idx], k, largest=True)  # align_metric 越大代表匹配越好
                matching_matrix[topk_idx, gt_idx] = True

            # 处理冲突：保留 align_metric 最高的
            anchor_matching_gt = matching_matrix.sum(1)
            if (anchor_matching_gt > 1).sum() > 0:
                conflict_indices = torch.where(anchor_matching_gt > 1)[0]
                for c_idx in conflict_indices:
                    matched_gts = torch.where(matching_matrix[c_idx])[0]
                    best_gt = matched_gts[torch.argmax(align_metric[c_idx, matched_gts])]
                    matching_matrix[c_idx] = False
                    matching_matrix[c_idx, best_gt] = True

            # 生成最终索引
            src_ind, tgt_ind = torch.where(matching_matrix)
            indices.append((src_ind, tgt_ind))

        return indices

@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 use_otm=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # === 保存开关配置 ===
        self.use_otm = use_otm

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # =================================================================
        # === 实例化动态 K 匹配器，并自动注册 O2M Loss 权重 ===
        # =================================================================
        if self.use_otm:
            self.dynamic_matcher = TaskAlignedDynamicKMatcher(alpha=1.0, beta=1.0, max_k=7)
            # 自动为 weight_dict 添加带有 '_o2m' 后缀的权重
            o2m_weight_dict = {}
            for k, v in self.weight_dict.items():
                o2m_weight_dict[f"{k}_o2m"] = v
            self.weight_dict.update(o2m_weight_dict)


    # === 【新增 2】密度 Loss 计算函数 ===
    def loss_density(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        【Focal Loss 版】Density Loss
        参考 Salience-DETR，解决正负样本（前景背景）极端不平衡问题。
        """
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        pred_logits = outputs['pred_density_map']  # [B, 1, H, W]

        # 1. 获取 Target 和 Weight (与之前保持一致)
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
            weight_map = kwargs['gt_weight_map']
        else:
            with torch.no_grad():
                # 注意：这里使用你之前定义的自适应生成函数
                target_map, weight_map = generate_targets_and_weights_adaptive(
                    targets, pred_logits.shape, pred_logits.device, 0.02
                )

        # 2. 计算 Focal Loss
        # Inputs: pred_logits (无 Sigmoid), target_map (0~1 float)
        # Alpha=0.25: 降低背景(负样本)权重
        # Gamma=2.0: 挖掘困难样本
        loss_focal = sigmoid_focal_loss(
            pred_logits,
            target_map,
            alpha=0.25,
            gamma=2.0,
            reduction='none'
        )

        # 3. 空间加权 (Apply Spatial Weight)
        # 对小目标区域再次加强 Loss
        loss_weighted = loss_focal * weight_map

        # 4. 归一化 (Normalization)
        # Salience-DETR 使用 num_pos (mask > 0.5 的像素数)
        # 这里推荐使用 target_map.sum() (Target Mass, 软正样本数)
        # 理由：你的 Target 是高斯/平顶分布的软标签，sum() 能反映目标的“总能量”。
        normalizer = target_map.sum()

        # DDP 同步
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()

        # 极小值保护
        normalizer = torch.clamp(normalizer, min=1.0)

        # 5. 最终 Loss
        # Focal Loss 的数值通常比 MSE 小很多 (数量级差异)
        # 建议不需要像 MSE 那样除以 2 或者乘 20，直接除以 normalizer 即可
        # 观察：如果 Loss 值过小(如 1e-3)，可以乘以一个系数 (e.g., * 5.0) 来平衡 Total Loss
        return {'loss_density': loss_weighted.sum() / normalizer}

    def loss_density_MSE2(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        【最终修正版】Weighted MSE + Target Mass Normalization
        既解决了大小物体平衡，又避免了背景稀释梯度。
        """
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        pred_logits = outputs['pred_density_map']

        # 1. 获取 Target 和 Weight
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
            weight_map = kwargs['gt_weight_map']
        else:
            with torch.no_grad():
                target_map, weight_map = generate_targets_and_weights_adaptive(
                    targets, pred_logits.shape, pred_logits.device, 0.02
                )

        # 2. Logits -> Sigmoid -> MSE
        pred_score = pred_logits.sigmoid()
        loss_pixel = F.mse_loss(pred_score, target_map, reduction='none')

        # 3. 加权 (你的小目标权重策略)
        loss_weighted = loss_pixel * weight_map

        # 4. 【核心修正】归一化：除以 Target 的总和 (Target Mass)
        # 含义：平均每个"正样本强度单位"的误差
        # 优点：
        #   1. 背景 Target=0，不占分母，不会稀释梯度。
        #   2. 大物体 Target Sum 大，Loss 被除得更多；小物体 Target Sum 小，Loss 保留更多。天然平衡！

        normalizer = target_map.sum()

        # DDP 同步
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()

        # 极小值保护 (防止全是背景时除以0)
        normalizer = torch.clamp(normalizer, min=1.0)

        # 5. 系数
        # 这种归一化方式下，Loss 大概在 0.01 ~ 0.05 级别
        # 建议乘 20.0，让最终 Loss 在 0.2 ~ 1.0 之间
        return {'loss_density': (loss_weighted.sum() / normalizer)/2}

    def loss_density_first(self, outputs, targets, indices, num_boxes, **kwargs):
        """计算 MSE Loss"""
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        src_map = outputs['pred_density_map']  # [B, 1, H, W]
        # 优先从 kwargs 获取预计算的 GT Map，避免重复计算
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
        # 动态生成 GT (不需要梯度)
        else:
            with torch.no_grad():
                target_map = generate_density_map_gt(
                    targets,
                    (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                    src_map.device,
                    sigma=2.0,
                )

        # 计算 MSE
        loss = F.mse_loss(src_map, target_map)
        return {'loss_density': loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,

            # === 【新增 3】注册 density ===
            'density': self.loss_density,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

        # ==========================================================
        # === 优化 1：将一致性合并逻辑提取为静态方法 (对齐 MS-DETR) ===
        # ==========================================================

    @staticmethod
    def indices_merge(indices_o2o, indices_o2m_raw):
        """
        合并 O2O 和 O2M 的匹配结果。
        策略：Union (并集)，且 O2M 优先级更高 (MS-DETR 逻辑)。
        """
        final_indices_o2m = []
        for i, (src_o2m, tgt_o2m) in enumerate(indices_o2m_raw):
            src_o2o, tgt_o2o = indices_o2o[i]

            # 1. 先放入 O2O (主匹配结果)
            # 使用字典 {Query_ID: GT_ID}
            match_dict = {s.item(): t.item() for s, t in zip(src_o2o, tgt_o2o)}

            # 2. 再放入 O2M (辅助匹配结果)
            # === 关键优化 ===
            # MS-DETR 逻辑是：temp_indices[o2m_idx] = o2m_gt
            # 这意味着如果同一个 Query 在 O2O 和 O2M 中都被选中，
            # 以 O2M 的分配为准 (通常两者是一样的，但如果有冲突，O2M 优先)
            for s, t in zip(src_o2m, tgt_o2m):
                match_dict[s.item()] = t.item()

            # 3. 重建 Tensor
            if len(match_dict) > 0:
                sorted_src = sorted(match_dict.keys())
                new_src = torch.as_tensor(sorted_src, dtype=torch.int64, device=src_o2m.device)
                new_tgt = torch.as_tensor([match_dict[s] for s in sorted_src], dtype=torch.int64,
                                          device=tgt_o2m.device)
            else:
                new_src = torch.as_tensor([], dtype=torch.int64, device=src_o2m.device)
                new_tgt = torch.as_tensor([], dtype=torch.int64, device=tgt_o2m.device)

            final_indices_o2m.append((new_src, new_tgt))

        return final_indices_o2m

    def forward(self, outputs, targets):
        # 【微调这里】加上 and 'o2m' not in k
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k and 'o2m' not in k}

        # 1. 主匹配：一对一 (One-to-One)
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 辅助匹配：一对多 (One-to-Many)
        indices_o2m = None
        # 计算 Loss
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        # 主 Loss
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # =================================================================
        # === B. 辅助分支 (O2M) 匹配与损失计算 (创新点三) ===
        # =================================================================
        if getattr(self, 'use_otm', False) and 'o2m_outputs' in outputs:
            o2m_outs = outputs['o2m_outputs']
            o2m_outs_clean = {k: v for k, v in o2m_outs.items() if k not in ['aux_outputs']}

            # 使用新写的动态 K 匹配器
            indices_o2m = self.dynamic_matcher(o2m_outs_clean, targets)

            # 计算 O2M 正样本数量
            num_boxes_o2m = sum(len(t[0]) for t in indices_o2m)
            num_boxes_o2m = torch.as_tensor([num_boxes_o2m], dtype=torch.float,
                                            device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_o2m)
            num_boxes_o2m = torch.clamp(num_boxes_o2m / get_world_size(), min=1).item()

            # 计算 O2M Loss (仅算分类和框，不算 density)
            o2m_loss_types = [l for l in self.losses if l not in ['masks', 'density']]

            for loss in o2m_loss_types:
                l_dict = self.get_loss(loss, o2m_outs_clean, targets, indices_o2m, num_boxes_o2m)
                # 赋予带有 _o2m 后缀的权重，比例默认为 1.0 (已在 init 中注册)
                l_dict = {f"{k}_o2m": v * self.weight_dict.get(f"{k}_o2m", 1.0) for k, v in l_dict.items()}
                losses.update(l_dict)

            # O2M 的深层监督 (Auxiliary Layers)
            if 'aux_outputs' in o2m_outs:
                for i, aux_outputs in enumerate(o2m_outs['aux_outputs']):
                    # 每一层独立匹配，保证特征不冲突
                    indices_o2m_aux = self.dynamic_matcher(aux_outputs, targets)
                    for loss in o2m_loss_types:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices_o2m_aux, num_boxes_o2m)
                        # RT-DETR 的 aux 命名规则通常是 loss_bbox_aux_0, 这里对应加上 _o2m
                        for k, v in l_dict.items():
                            weight_key = f"{k}_aux_{i}_o2m"
                            val = v * self.weight_dict.get(weight_key, self.weight_dict.get(f"{k}_o2m", 1.0))
                            losses.update({weight_key: val})

        # Decoder Aux Loss (Intermediate layers)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' or loss == 'density': continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # DN Loss
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes_dn = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks' or loss == 'density': continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, num_boxes_dn, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices


def generate_density_map_gt(targets, feat_shape, device, sigma=None):  # sigma 参数设为 None 或默认值
    B, H, W = feat_shape
    density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y = grid_y.unsqueeze(0)
    grid_x = grid_x.unsqueeze(0)

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']  # [N, 4]
        N = len(boxes)

        # 1. 计算自适应 Sigma
        # 宽和高转为特征图尺度
        w_feat = boxes[:, 2] * W
        h_feat = boxes[:, 3] * H

        # 核心公式：sigma = max(1, min(w, h) * ratio)
        # 比如取宽高中较小值的 1/2 作为半径，再除以 3 得到 sigma (3-sigma原则)
        # 或者直接简化为：宽高最小值的 1/3
        # 加上 clamp(min=0.5) 防止极小目标 sigma 接近 0 导致数值异常
        adaptive_sigmas = torch.clamp(torch.min(w_feat, h_feat) / 2.0, min=0.8)  # [N]
        adaptive_sigmas = adaptive_sigmas.view(N, 1, 1)

        # 2. 权重 (保持你原有的逻辑)
        areas = boxes[:, 2] * boxes[:, 3]
        scale_factor = 100.0
        weights = 1.0 + 4.0 * torch.exp(-areas * scale_factor)
        weights = weights.view(N, 1, 1)

        # 3. 中心点
        cx = (boxes[:, 0] * W).view(N, 1, 1)
        cy = (boxes[:, 1] * H).view(N, 1, 1)

        # 4. 计算高斯 (注意 sigma 现在是变量 [N, 1, 1])
        dist_sq = (grid_x + 0.5 - cx) ** 2 + (grid_y + 0.5 - cy) ** 2

        # 广播机制会自动处理 adaptive_sigmas
        gaussian = torch.exp(-dist_sq / (2 * adaptive_sigmas ** 2))

        weighted_gaussian = gaussian * weights

        if N > 0:
            val, _ = weighted_gaussian.max(dim=0)
            density_map[i, 0] = val

    return density_map


def generate_targets_and_weights_adaptive(targets, feat_shape, device, threshold=0.02):
    """
    【S3 最终版】尺度自适应 Box-Aware Centerness
    核心创新：
    小目标像素 0.0025
    中目标 0.0225
    1. 小目标使用极小的 beta (如 0.2)，使其高分区域非常宽（平顶分布）。
    2. 稍大的目标使用正常的 beta (如 0.8)，保持定位精度。
    """
    B, _, H, W = feat_shape
    target_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    weight_map = torch.ones((B, 1, H, W), dtype=torch.float32, device=device)
    total_valid_objects = 0

    # ... (网格生成代码同前，省略) ...
    # 构造网格
    y_range = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    x_range = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y_flat = grid_y.flatten()
    grid_x_flat = grid_x.flatten()

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']
        areas = boxes[:, 2] * boxes[:, 3]

        # 1. 过滤大目标 (保持不变)
        is_valid_scale = areas < threshold
        if not is_valid_scale.any(): continue

        valid_boxes = boxes[is_valid_scale]
        valid_areas = areas[is_valid_scale]
        N = len(valid_boxes)

        # 坐标转换 ... (同前，省略)
        cx = valid_boxes[:, 0] * W
        cy = valid_boxes[:, 1] * H
        w_half = (valid_boxes[:, 2] * W) / 2.0
        h_half = (valid_boxes[:, 3] * H) / 2.0
        x1 = (cx - w_half).view(N, 1)
        y1 = (cy - h_half).view(N, 1)
        x2 = (cx + w_half).view(N, 1)
        y2 = (cy + h_half).view(N, 1)

        # ... (计算 l, r, t, b 同前) ...
        l = grid_x_flat[None, :] - x1
        r = x2 - grid_x_flat[None, :]
        t = grid_y_flat[None, :] - y1
        b = y2 - grid_y_flat[None, :]
        is_in_box = (l > 0) & (r > 0) & (t > 0) & (b > 0)

        # === 2. 计算基础 Centerness ===
        min_lr = torch.min(l, r)
        max_lr = torch.max(l, r)
        min_tb = torch.min(t, b)
        max_tb = torch.max(t, b)
        # 计算乘积
        product = (min_lr / max_lr.clamp(min=1e-6)) * (min_tb / max_tb.clamp(min=1e-6))

        # 【关键修正】先 clamp 到 0，再开根号！
        # 防止部分在内部分在外导致的负数乘积
        raw_centerness = torch.sqrt(product.clamp(min=0))

        # === [核心修改] 3. 动态 Beta 计算 ===
        # 逻辑：面积越小，beta 越小 (分布越平/越宽)
        # 设面积阈值：0.005 (极小) -> beta=0.2
        #             0.04  (中等) -> beta=1.0
        # 建立一个线性映射或分段映射

        # 归一化面积到 0~1 之间 (相对于 0.04)
        area_ratio = valid_areas / threshold

        # 映射公式：beta = 0.2 + 0.8 * area_ratio
        # 极小物体(area~0): beta -> 0.2 (超级平顶，像个方块)
        # 较大物体(area~0.04): beta -> 1.0 (正常金字塔)

        box_betas = 0.2 + 0.8 * area_ratio
        box_betas = torch.clamp(box_betas, min=0.2, max=1.0)

        # 广播到 [N, 1]
        box_betas = box_betas.view(N, 1)

        # 应用动态衰减
        # Pow 操作会自动广播: [N, H*W] ^ [N, 1]
        final_target = torch.pow(raw_centerness, box_betas)

        # Masking
        final_target = final_target * is_in_box.float()

        # === 4. 权重计算 (保持之前的强力加权) ===
        scale_factor = 100.0
        box_weights = 1.0 + 0 * torch.exp(-valid_areas * scale_factor)
        box_weights = box_weights.view(N, 1) * is_in_box.float()

        # === 5. 聚合 ===
        if N > 0:
            max_target, _ = final_target.max(dim=0)
            target_map[i, 0] = max_target.view(H, W)

            max_weight, _ = box_weights.max(dim=0)
            current_weight_plane = max_weight.view(H, W)
            weight_map[i, 0] = torch.maximum(weight_map[i, 0], current_weight_plane)

    return target_map, weight_map


def generate_targets_and_weights(targets, feat_shape, device, decay_beta=0.5):
    """
    【修复 NaN Bug 版】全向量化生成 Box-Aware Density & Weight
    修复点：在 sqrt 之前对负数进行 clamp(min=0)，防止 Box 外部点产生 NaN。
    """
    B, H, W = feat_shape
    density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    weight_map = torch.ones((B, 1, H, W), dtype=torch.float32, device=device)
    total_valid_objects = 0

    # 构造网格
    y_range = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    x_range = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y_flat = grid_y.flatten()
    grid_x_flat = grid_x.flatten()

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']
        areas = boxes[:, 2] * boxes[:, 3]

        # 尺度过滤
        is_valid_scale = areas < 0.04
        if not is_valid_scale.any():
            continue

        valid_boxes = boxes[is_valid_scale]
        valid_areas = areas[is_valid_scale]
        N = len(valid_boxes)

        # 必须检查 N > 0
        if N == 0: continue

        total_valid_objects += N

        # 坐标变换
        cx = valid_boxes[:, 0] * W
        cy = valid_boxes[:, 1] * H
        w_half = (valid_boxes[:, 2] * W) / 2.0
        h_half = (valid_boxes[:, 3] * H) / 2.0

        x1 = (cx - w_half).view(N, 1)
        y1 = (cy - h_half).view(N, 1)
        x2 = (cx + w_half).view(N, 1)
        y2 = (cy + h_half).view(N, 1)

        # 计算距离
        l = grid_x_flat[None, :] - x1
        r = x2 - grid_x_flat[None, :]
        t = grid_y_flat[None, :] - y1
        b = y2 - grid_y_flat[None, :]

        is_in_box = (l > 0) & (r > 0) & (t > 0) & (b > 0)

        # Part A: Target Map
        min_lr = torch.min(l, r)
        max_lr = torch.max(l, r)
        min_tb = torch.min(t, b)
        max_tb = torch.max(t, b)

        # === 【关键修复】 ===
        # 计算乘积
        product = (min_lr / max_lr.clamp(min=1e-6)) * (min_tb / max_tb.clamp(min=1e-6))

        # 必须先 clamp 到 0，再开根号！
        # 否则 Box 外部的点是负数，sqrt(负数) = NaN，NaN * 0 还是 NaN
        centerness = torch.sqrt(product.clamp(min=0))

        centerness = torch.pow(centerness, decay_beta)
        centerness_scores = centerness * is_in_box.float()

        # Part B: Weight Map
        scale_factor = 100.0
        box_weight_scalars = 1.0 + 3.0 * torch.exp(-valid_areas * scale_factor)
        box_weight_maps = box_weight_scalars.view(N, 1) * is_in_box.float()

        # Part C: 聚合
        if N > 0:
            max_conf, _ = centerness_scores.max(dim=0)
            density_map[i, 0] = max_conf.view(H, W)

            max_weights, _ = box_weight_maps.max(dim=0)
            current_weight_plane = max_weights.view(H, W)
            weight_map[i, 0] = torch.maximum(weight_map[i, 0], current_weight_plane)

    return density_map, weight_map, total_valid_objects


def generate_density_map_gt_2(targets, feat_shape, device, sigma=None):
    """
    【S3 专属版】生成 0~1 的高斯密度图 GT。
    关键特性：过滤掉大目标 (Large Objects)，使其 Target=0。
    """
    B, H, W = feat_shape
    density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    # 1. 坐标网格
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y = grid_y.unsqueeze(0)  # [1, H, W]
    grid_x = grid_x.unsqueeze(0)

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']  # [N, 4] (cx, cy, w, h)
        areas = boxes[:, 2] * boxes[:, 3]

        # === [核心修改] 尺度过滤 (Scale Filtering) ===
        # 过滤掉面积大于 0.04 (约 128x128 像素) 的大目标
        # 这些大目标在 S3 密度图上被视为背景 (Target=0)
        is_valid_scale = areas < 0.04

        if not is_valid_scale.any():
            continue

        # 只处理中小目标
        valid_boxes = boxes[is_valid_scale]
        valid_areas = areas[is_valid_scale]
        N = len(valid_boxes)

        # 2. 转换到特征图尺度
        w_feat = valid_boxes[:, 2] * W
        h_feat = valid_boxes[:, 3] * H
        cx = valid_boxes[:, 0] * W
        cy = valid_boxes[:, 1] * H

        # 3. 基础 Sigma & 小目标补偿
        base_sigma = torch.clamp(torch.min(w_feat, h_feat) / 2.0, min=1.0)  # [N]
        is_very_small = valid_areas < 0.003

        adaptive_sigmas = torch.where(is_very_small, base_sigma * 1.5 + 1.0, base_sigma)
        adaptive_sigmas = adaptive_sigmas.view(N, 1, 1)

        # 4. 向量化计算高斯
        cx = cx.view(N, 1, 1)
        cy = cy.view(N, 1, 1)

        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        gaussian = torch.exp(-dist_sq / (2 * adaptive_sigmas ** 2))  # [N, H, W]

        # 5. Max 聚合
        if N > 0:
            val, _ = gaussian.max(dim=0)
            density_map[i, 0] = val

    return density_map


# === 【新增 1】GT 生成工具函数 (加在 SetCriterion 类外面) ===
def generate_density_map_gt_3(targets, feat_shape, device, sigma=None):
    """
    【修正版】生成 0~1 的高斯密度图 GT，并包含小目标 Sigma 补偿。
    注意：不再包含 area weight，权重将移至 Loss 计算中。
    """
    B, H, W = feat_shape
    density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    # 1. 坐标网格
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y = grid_y.unsqueeze(0)  # [1, H, W]
    grid_x = grid_x.unsqueeze(0)

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']  # [N, 4] (cx, cy, w, h) 归一化坐标
        N = len(boxes)

        # 2. 转换到特征图尺度
        w_feat = boxes[:, 2] * W
        h_feat = boxes[:, 3] * H
        cx = boxes[:, 0] * W
        cy = boxes[:, 1] * H

        # 3. 基础 Sigma (3-sigma 原则，半径约为 min(w,h)/2)
        # 加上 clamp 防止过小
        base_sigma = torch.clamp(torch.min(w_feat, h_feat) / 2.0, min=1.0)  # [N]

        # === 创新点：小目标 Sigma 空间补偿 ===
        # 逻辑：如果物体很小，我们人为“虚胖”它的高斯核，
        # 让它在特征图上占据更多像素，增加被 Query 选中的容错率。
        # 阈值：32像素对应 stride 8 的特征图上是 4。4*4=16。
        # 相对面积阈值：(32/640)^2 ≈ 0.0025
        areas = boxes[:, 2] * boxes[:, 3]
        is_small = areas < 0.003  # 稍微放宽一点阈值

        # 对小目标：Sigma 放大 1.5 倍 + 1.0 的偏置
        adaptive_sigmas = torch.where(is_small, base_sigma * 1.5 + 1.0, base_sigma)
        adaptive_sigmas = adaptive_sigmas.view(N, 1, 1)

        # 4. 向量化计算高斯 (Target 峰值为 1.0)
        # cx, cy 广播
        cx = cx.view(N, 1, 1)
        cy = cy.view(N, 1, 1)

        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        gaussian = torch.exp(-dist_sq / (2 * adaptive_sigmas ** 2))  # [N, H, W]

        # 5. Max 聚合
        if N > 0:
            val, _ = gaussian.max(dim=0)
            density_map[i, 0] = val

    return density_map


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
