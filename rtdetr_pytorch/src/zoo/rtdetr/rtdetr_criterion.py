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
# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register

from .visualize import visualize_boxes


# ==========================================
# === 创新点三：密度引导的一对多匹配器 ===
# ==========================================
class DensityGuidedMatcher(nn.Module):
    """
    Density-Guided One-to-Many Matcher for Auxiliary Training
    """

    def __init__(self, weight_dict, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma

        # 代价系数
        self.cost_class = 2
        self.cost_bbox = 5
        self.cost_giou = 2
        self.cost_density = 1.0

    @torch.no_grad()
    def forward(self, outputs, targets, density_map):
        """
        density_map: [B, 1, H, W] 这里传入的必须是 GT 密度图！
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 展平预测
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # 2. 准备 GT
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. 计算基础 Matching Cost
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 4. 遍历 Batch 处理
        indices = []
        cur_idx = 0

        # === 【修改：移除动态最大值归一化，改为全局缩放】 ===
        # 你的 GT 生成公式最大值约为 5.0 (1.0 + 4.0)。
        # 为了让 density 在 0~1 之间且保留大小物体差异，除以固定值 5.0
        density_map_norm = density_map / 5.0

        for i in range(bs):
            tgt_ids_i = targets[i]["labels"]
            tgt_bbox_i = targets[i]["boxes"]
            num_gt = len(tgt_ids_i)

            if num_gt == 0:
                indices.append((
                    torch.as_tensor([], dtype=torch.int64, device=out_bbox.device),
                    torch.as_tensor([], dtype=torch.int64, device=out_bbox.device)
                ))
                cur_idx += num_queries
                continue

            # 切片 Cost
            c_class = cost_class[cur_idx: cur_idx + num_queries, :num_gt]
            c_bbox = cost_bbox[cur_idx: cur_idx + num_queries, :num_gt]
            c_giou = cost_giou[cur_idx: cur_idx + num_queries, :num_gt]

            # === 计算 Density Cost (使用 GT Density) ===
            current_density = density_map_norm[i].unsqueeze(0)  # [1, 1, H, W]
            current_centers = outputs["pred_boxes"][i, :, :2]

            # grid_sample 需要 [-1, 1] 坐标
            grid_coords = current_centers.view(1, num_queries, 1, 2) * 2.0 - 1.0

            # 采样
            sampled_density = F.grid_sample(current_density, grid_coords, align_corners=False).view(num_queries)

            # 密度越大，Cost 越小
            c_density = -sampled_density.unsqueeze(1).repeat(1, num_gt)

            # === 总 Cost ===
            C = self.cost_bbox * c_bbox + \
                self.cost_class * c_class + \
                self.cost_giou * c_giou + \
                self.cost_density * c_density

            # === Scale-Aware Top-K 选择 ===
            tgt_areas = tgt_bbox_i[:, 2] * tgt_bbox_i[:, 3]
            is_small = tgt_areas < 0.005

            # === 【修正点2】Top-K 冲突消解 ===
            # 问题：两个物体靠得近，同一个 Query 可能同时进入两个 GT 的 Top-K
            # 解决：使用 Matrix 记录匹配，对于冲突的 Query，只保留 Cost 最小的那个 GT

            # 1. 建立匹配矩阵 [Num_Queries, Num_GT]
            matching_matrix = torch.zeros_like(C, dtype=torch.bool)

            for gt_idx in range(num_gt):
                k = 6 if is_small[gt_idx] else 4
                k = min(k, num_queries)
                _, topk_indices = torch.topk(C[:, gt_idx], k, largest=False)
                matching_matrix[topk_indices, gt_idx] = True

            # 2. 检查冲突：一个 Query 匹配了 >1 个 GT
            anchor_matching_gt = matching_matrix.sum(1)  # [Num_Queries]

            if (anchor_matching_gt > 1).sum() > 0:
                conflict_indices = torch.where(anchor_matching_gt > 1)[0]
                for c_idx in conflict_indices:
                    # 找到该 Query 匹配的所有 GT
                    matched_gts = torch.where(matching_matrix[c_idx])[0]
                    # 找到这些 GT 中，该 Query 对应的 Cost 最小的那个
                    best_gt = matched_gts[torch.argmin(C[c_idx, matched_gts])]

                    # 修正：先全清空，再赋值最佳
                    matching_matrix[c_idx] = False
                    matching_matrix[c_idx, best_gt] = True

            # 3. 生成最终索引
            src_ind, tgt_ind = torch.where(matching_matrix)
            indices.append((src_ind, tgt_ind))

            cur_idx += num_queries

        return indices


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
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 use_density_aux_loss=False):
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
        self.use_density_aux_loss = use_density_aux_loss

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # 初始化辅助匹配器
        self.aux_matcher = DensityGuidedMatcher(weight_dict, alpha, gamma)

    def loss_density3(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        【最终优化版】Weighted QFL + Box-Aware Labels + DDP Norm
        """
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        src_map = outputs['pred_density_map']  # [B, 1, H, W]
        device = src_map.device

        # 1. 数值稳定性保护
        src_map = torch.clamp(src_map, min=1e-6, max=1.0 - 1e-6)

        # 2. 获取 Target Map 和 Weight Map (一步完成)
        # 如果 kwargs 里有缓存（例如在 Dataset 里算好的），直接用
        # 否则在线计算（现在这个在线计算非常快）
        if 'gt_density_map' in kwargs and 'gt_weight_map' in kwargs:
            target_map = kwargs['gt_density_map']
            weight_map = kwargs['gt_weight_map']
            # 注意：如果从外部传，需要确保外部也传了 num_valid_objects，否则这里得重新算或估算
            # 鉴于你是在线算，这里主要走 else 分支
            total_valid_objects = kwargs.get('num_valid_objects', 1.0)
        else:
            with torch.no_grad():
                target_map, weight_map, total_valid_objects = generate_targets_and_weights(
                    targets,
                    (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                    device,
                    decay_beta=0.5  # 推荐 0.3-0.5，让小目标高分区域更宽
                )

        # 3. 计算 QFL (Weighted)
        beta = 2.0
        scale = torch.abs(target_map - src_map) ** beta
        loss_bce = F.binary_cross_entropy(src_map, target_map, reduction='none')

        # 分子：加权求和
        loss_unnormalized = (weight_map * scale * loss_bce).sum()

        # 4. DDP 全局归一化
        # 将本地 count 转为 Tensor
        num_valid_objects = torch.tensor(total_valid_objects, dtype=torch.float, device=device)

        # 多卡同步求平均
        global_avg_objects = reduce_mean(num_valid_objects)

        # 极小值保护
        pos_normalizer = torch.clamp(global_avg_objects, min=1.0)

        # 5. 返回最终 Loss
        # 建议系数 2.0
        return {'loss_density': (loss_unnormalized / pos_normalizer)}

    def loss_density3(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        【适配 Sigmoid + 尺度约束版】计算加权 BCE Loss。
        Target: 0~1 (大目标区域为 0)
        Weight: 1.0 (背景/大目标) ~ 4.0 (小目标)
        """
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        src_map = outputs['pred_density_map']  # [B, 1, H, W] (已经是 0~1)

        # 1. 获取 Target Map (0~1, 大目标已过滤为 0)
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
        else:
            with torch.no_grad():
                target_map = generate_density_map_gt_2(
                    targets,
                    (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                    src_map.device
                )

        # 2. 动态生成 Pixel-wise Loss 权重图
        B, C, H, W = src_map.shape
        weight_map = torch.ones_like(target_map)  # 默认为 1.0 (背景权重)

        for i in range(B):
            if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
                continue

            boxes = targets[i]['boxes']
            areas = boxes[:, 2] * boxes[:, 3]

            # === [同步修改] 尺度过滤 ===
            # 在计算权重时，同样忽略大目标。
            # 让大目标区域的权重保持默认的 1.0 (背景权重)，迫使模型将其预测为 0。
            # 阈值必须与 GT 生成函数保持一致 (0.04)
            is_valid_scale = areas < 0.04

            if not is_valid_scale.any():
                continue

            valid_boxes = boxes[is_valid_scale]
            valid_areas = areas[is_valid_scale]

            # 计算每个框的权重值
            scale_factor = 100.0
            # 权重公式：小目标权重高 (~4.0)，大目标(如果没过滤)权重低 (~1.0)
            box_weights = 1.0 + 3.0 * torch.exp(-valid_areas * scale_factor)

            # 将权重填入对应的框区域
            feat_boxes = valid_boxes * torch.tensor([W, H, W, H], device=src_map.device)
            feat_boxes = box_cxcywh_to_xyxy(feat_boxes).long()

            # 限制坐标在图像内
            feat_boxes[:, 0::2].clamp_(0, W - 1)
            feat_boxes[:, 1::2].clamp_(0, H - 1)

            for k in range(len(valid_boxes)):
                x1, y1, x2, y2 = feat_boxes[k]
                w_val = box_weights[k]
                # 处理重叠区域，保留最大权重
                current_roi = weight_map[i, 0, y1:y2 + 1, x1:x2 + 1]
                weight_map[i, 0, y1:y2 + 1, x1:x2 + 1] = torch.maximum(current_roi, w_val)

        # 3. 计算加权 BCE Loss
        loss = F.binary_cross_entropy(src_map, target_map, reduction='none')

        # 应用权重并求平均
        loss = (loss * weight_map).mean()

        return {'loss_density': loss}

    # === 【新增 2】密度 Loss 计算函数 ===
    def loss_density2(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        【适配 Sigmoid 版】计算加权 BCE Loss。
        Target: 0~1
        Input: 0~1 (Sigmoid processed)
        Weight: 1.0 (Background) ~ 5.0 (Small Object) applied to Loss
        """
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        src_map = outputs['pred_density_map']  # [B, 1, H, W] (已经是 0~1)

        # 1. 获取 Target Map (0~1)
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
        else:
            with torch.no_grad():
                # 使用修正后的 GT 生成函数
                target_map = generate_density_map_gt_2(
                    targets,
                    (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                    src_map.device
                )

        # 2. 动态生成 Pixel-wise Loss 权重图
        B, C, H, W = src_map.shape
        weight_map = torch.ones_like(target_map)  # 默认为 1.0 (背景权重)

        for i in range(B):
            if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
                continue

            boxes = targets[i]['boxes']
            # 计算每个框的权重值
            areas = boxes[:, 2] * boxes[:, 3]
            scale_factor = 100.0
            # 权重公式：小目标权重高 (~5.0)，大目标权重低 (~1.0)
            box_weights = 1.0 + 3.0 * torch.exp(-areas * scale_factor)

            # 将权重填入对应的框区域
            # 将归一化坐标转为特征图坐标
            feat_boxes = boxes * torch.tensor([W, H, W, H], device=src_map.device)
            feat_boxes = box_cxcywh_to_xyxy(feat_boxes).long()

            # 限制坐标在图像内
            feat_boxes[:, 0::2].clamp_(0, W - 1)
            feat_boxes[:, 1::2].clamp_(0, H - 1)

            for k in range(len(boxes)):
                x1, y1, x2, y2 = feat_boxes[k]
                w_val = box_weights[k]
                # 处理重叠区域，保留最大权重 (即保留对小目标的关注)
                current_roi = weight_map[i, 0, y1:y2 + 1, x1:x2 + 1]
                weight_map[i, 0, y1:y2 + 1, x1:x2 + 1] = torch.maximum(current_roi, w_val)

        # 3. 计算加权 BCE Loss
        # 注意：src_map 已经是 Sigmoid 过的，所以用 binary_cross_entropy
        # reduction='none' 以便应用权重
        loss = F.binary_cross_entropy(src_map, target_map, reduction='none')

        # 应用权重并求平均
        loss = (loss * weight_map).mean()

        return {'loss_density': loss}

    def loss_density(self, outputs, targets, indices, num_boxes, **kwargs):
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
        return {'loss_density': (loss_weighted.sum() / normalizer)/5}

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
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # 1. 主匹配：一对一 (One-to-One)
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 辅助匹配：一对多 (One-to-Many)
        indices_o2m = None
        # 定义一个变量保存 GT Density Map，供 Loss 使用，避免重复计算
        gt_density_map = None

        if self.use_density_aux_loss and 'o2m_outputs' in outputs:
            # === 关键：使用 O2M 输出进行辅助匹配 ===
            o2m_outputs = outputs['o2m_outputs']
            # === 【关键修改：获取形状参考】 ===
            # === 更紧凑的 Reference Map 获取逻辑 ===
            ref_map = outputs.get('pred_density_map', outputs.get('s3_shape_ref', None))
            if ref_map is None:
                raise ValueError("No reference map (pred_density_map or s3_shape_ref) found!")
            # === 生成 GT Density Map 用于匹配 (修正点1) ===
            with torch.no_grad():
                gt_density_map = generate_density_map_gt(
                    targets,
                    (ref_map.shape[0], ref_map.shape[2], ref_map.shape[3]),
                    ref_map.device,
                    sigma=1.0
                )

            # 将 O2M 输出传给匹配器 (使用 O2M Head 的 logits)
            indices_o2m_raw = self.aux_matcher(o2m_outputs, targets, gt_density_map)

            # === 调用优化后的一致性合并 ===
            indices_o2m = self.indices_merge(indices, indices_o2m_raw)

        # 计算 Loss
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        # 将 GT Density Map 传入 kwargs，供 loss_density 使用
        kwargs_main = {}
        if gt_density_map is not None:
            kwargs_main['gt_density_map'] = gt_density_map

        # 主 Loss
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs_main)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # 辅助 Loss (Density Guided)
        if indices_o2m is not None:
            aux_weight_scale = 0.2
            aux_loss_types = ['vfl', 'boxes'] if 'vfl' in self.losses else ['labels', 'boxes']
            if 'focal' in self.losses: aux_loss_types = ['focal', 'boxes']

            for loss in aux_loss_types:
                if loss not in self.losses: continue
                # === 关键：传入 outputs['o2m_outputs'] ===
                # 这样梯度会回传给 dec_score_head_o2m，而不会污染主分支
                l_dict = self.get_loss(loss, outputs['o2m_outputs'], targets, indices_o2m, num_boxes)
                l_dict = {f"{k}_o2m": v * self.weight_dict.get(k, 1.0) * aux_weight_scale for k, v in l_dict.items()}
                losses.update(l_dict)

            # === 6. 辅助分支 Loss (O2M - Deep Supervision) ===
            # 这里是之前缺失的部分！
            if 'aux_outputs' in outputs['o2m_outputs']:
                for i, aux_outputs in enumerate(outputs['o2m_outputs']['aux_outputs']):
                    # 为了严谨，每一层应该重新匹配。
                    # 复用 gt_density_map (假设 spatial shape 没变，或者自动适配)
                    # 如果 decoder 中间层 shape 不一样，这里 generate 可能会有维度问题，
                    # 但 RT-DETR decoder 层间 shape 通常是一样的。

                    indices_o2m_aux_raw = self.aux_matcher(aux_outputs, targets, gt_density_map)
                    indices_o2m_aux = self.indices_merge(indices, indices_o2m_aux_raw)

                    for loss in aux_loss_types:
                        if loss not in self.losses: continue
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices_o2m_aux, num_boxes)
                        # 注意命名和权重
                        l_dict = {f"{k}_aux_{i}_o2m": v * self.weight_dict.get(k, 1.0) * aux_weight_scale for k, v
                                  in l_dict.items()}
                        losses.update(l_dict)

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
        box_weights = 1.0 + 10.0 * torch.exp(-valid_areas * scale_factor)
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
