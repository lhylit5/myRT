"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

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
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
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

        # === 【修正点1】归一化密度图 ===
        # 你的 weights 最大有 5.0，如果不归一化，Cost 很容易爆炸
        # 按每张图的最大值进行归一化，保持在 0-1 之间
        max_vals = density_map.flatten(2).max(2)[0][..., None, None]  # [B, 1, 1, 1]
        density_map_norm = density_map / (max_vals + 1e-6)

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
                k = 6 if is_small[gt_idx] else 2
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

@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80, use_density_aux_loss=False):
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

    # === 【新增 2】密度 Loss 计算函数 ===
    def loss_density(self, outputs, targets, indices, num_boxes, **kwargs):
        """计算 MSE Loss"""
        if 'pred_density_map' not in outputs:
            return {'loss_density': torch.tensor(0.0).to(outputs['pred_boxes'].device)}

        src_map = outputs['pred_density_map']  # [B, 1, H, W]
        # 优先从 kwargs 获取预计算的 GT Map，避免重复计算
        if 'gt_density_map' in kwargs:
            target_map = kwargs['gt_density_map']
        # 动态生成 GT (不需要梯度)
        with torch.no_grad():
            target_map = generate_density_map_gt(
                targets,
                (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                src_map.device,
                sigma=2.0  # 对于 Stride=8 (S3)，sigma=2.0 比较合适
            )

        # 计算 MSE
        loss = F.mse_loss(src_map, target_map)
        return {'loss_density': loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
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

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
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
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
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

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

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

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # 1. 主匹配：一对一 (One-to-One)
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 辅助匹配：一对多 (One-to-Many)
        indices_o2m = None
        # 定义一个变量保存 GT Density Map，供 Loss 使用，避免重复计算
        gt_density_map = None

        if self.use_density_aux_loss and 'pred_density_map' in outputs:
            # === 生成 GT Density Map 用于匹配 (修正点1) ===
            src_map = outputs['pred_density_map']
            with torch.no_grad():
                gt_density_map = generate_density_map_gt(
                    targets,
                    (src_map.shape[0], src_map.shape[2], src_map.shape[3]),
                    src_map.device,
                    sigma=2.0
                )

            # 传入 GT Map 进行匹配
            indices_o2m_raw = self.aux_matcher(outputs_without_aux, targets, gt_density_map)

            # === 【修正点3：强制一致性 (Consistency Enforcement)】 ===
            # 策略：如果主匹配 (O2O) 认为 Query A 是 GT B，那么辅助匹配 (O2M) 也必须包含这个配对
            final_indices_o2m = []
            for i, (src_o2m, tgt_o2m) in enumerate(indices_o2m_raw):
                src_o2o, tgt_o2o = indices[i]

                # 使用字典合并：O2O 的结果覆盖/添加到 O2M 中
                match_dict = {s.item(): t.item() for s, t in zip(src_o2m, tgt_o2m)}
                for s, t in zip(src_o2o, tgt_o2o):
                    match_dict[s.item()] = t.item()

                # 重建 Tensor
                if len(match_dict) > 0:
                    sorted_src = sorted(match_dict.keys())
                    new_src = torch.as_tensor(sorted_src, dtype=torch.int64, device=src_o2m.device)
                    new_tgt = torch.as_tensor([match_dict[s] for s in sorted_src], dtype=torch.int64,
                                              device=tgt_o2m.device)
                else:
                    new_src = torch.as_tensor([], dtype=torch.int64, device=src_o2m.device)
                    new_tgt = torch.as_tensor([], dtype=torch.int64, device=tgt_o2m.device)

                final_indices_o2m.append((new_src, new_tgt))

            indices_o2m = final_indices_o2m

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
                l_dict = self.get_loss(loss, outputs, targets, indices_o2m, num_boxes)
                l_dict = {f"{k}_o2m": v * self.weight_dict.get(k, 1.0) * aux_weight_scale for k, v in l_dict.items()}
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
    # def forward(self, outputs, targets):
    #     """ This performs the loss computation.
    #     Parameters:
    #          outputs: dict of tensors, see the output specification of the model for the format
    #          targets: list of dicts, such that len(targets) == batch_size.
    #                   The expected keys in each dict depends on the losses applied, see each loss' doc
    #     """
    #     outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
    #
    #     # Retrieve the matching between the outputs of the last layer and the targets
    #     indices = self.matcher(outputs_without_aux, targets)
    #     # 2. === 创新点3：密度引导的一对多匹配 (可配置开关) ===
    #     indices_o2m = None
    #     # 定义一个变量保存 GT Density Map，供 Loss 使用，避免重复计算
    #     gt_density_map = None
    #     # 只有当【开关开启】且【存在密度图】时，才执行辅助匹配
    #     if self.use_density_aux_loss and 'pred_density_map' in outputs:
    #         indices_o2m = self.aux_matcher(outputs_without_aux, targets, outputs['pred_density_map'])
    #     first_indices = indices
    #
    #     # Compute the average number of target boxes accross all nodes, for normalization purposes
    #     num_boxes = sum(len(t["labels"]) for t in targets)
    #     num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    #     if is_dist_available_and_initialized():
    #         torch.distributed.all_reduce(num_boxes)
    #     num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    #
    #     # Compute all the requested losses
    #     losses = {}
    #     for loss in self.losses:
    #         l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
    #         l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
    #         losses.update(l_dict)
    #
    #     # 4. === 计算辅助 Loss (One-to-Many) ===
    #     if indices_o2m is not None:
    #         # 【关键修改】降低权重，防止梯度冲突破坏主分支
    #         # 建议设置为 0.1 ~ 0.5 之间。
    #         # 这相当于告诉模型：“主要听一对一的，但也要适当参考密度图的建议”
    #         aux_weight_scale = 0.2
    #
    #         aux_loss_types = ['vfl', 'boxes'] if 'vfl' in self.losses else ['labels', 'boxes']
    #         if 'focal' in self.losses: aux_loss_types = ['focal', 'boxes']
    #
    #         for loss in aux_loss_types:
    #             if loss not in self.losses: continue
    #             l_dict = self.get_loss(loss, outputs, targets, indices_o2m, num_boxes)
    #             # 乘上 aux_weight_scale
    #             l_dict = {f"{k}_o2m": v * self.weight_dict.get(k, 1.0) * aux_weight_scale for k, v in
    #                       l_dict.items()}
    #             losses.update(l_dict)
    #
    #     # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    #     if 'aux_outputs' in outputs:
    #         for i, aux_outputs in enumerate(outputs['aux_outputs']):
    #             indices = self.matcher(aux_outputs, targets)
    #             for loss in self.losses:
    #                 if loss == 'masks':
    #                     # Intermediate masks losses are too costly to compute, we ignore them.
    #                     continue
    #                 kwargs = {}
    #                 if loss == 'labels':
    #                     # Logging is enabled only for the last layer
    #                     kwargs = {'log': False}
    #
    #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
    #                 l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
    #                 l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
    #                 losses.update(l_dict)
    #
    #     # In case of cdn auxiliary losses. For rtdetr
    #     if 'dn_aux_outputs' in outputs:
    #         assert 'dn_meta' in outputs, ''
    #         indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
    #         num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
    #
    #         for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
    #             # indices = self.matcher(aux_outputs, targets)
    #             for loss in self.losses:
    #                 if loss == 'masks':
    #                     # Intermediate masks losses are too costly to compute, we ignore them.
    #                     continue
    #                 kwargs = {}
    #                 if loss == 'labels':
    #                     # Logging is enabled only for the last layer
    #                     kwargs = {'log': False}
    #
    #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
    #                 l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
    #                 l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
    #                 losses.update(l_dict)
    #
    #     return losses, first_indices

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
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices


# === 【新增 1】GT 生成工具函数 (加在 SetCriterion 类外面) ===
def generate_density_map_gt(targets, feat_shape, device, sigma=2.0):
    """
    【高性能版】生成【面积加权】的高斯密度图 GT。
    效果与 generate_density_map_gt 严格一致 (Max聚合 + 无归一化)。
    """
    B, H, W = feat_shape
    density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    # 1. 预先生成网格坐标 [1, H, W]
    # 注意：meshgrid 的 indexing='ij' 对应 (y, x)，与你的 grid_y, grid_x 逻辑一致
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    # 扩展维度以便广播: [1, H, W]
    grid_y = grid_y.unsqueeze(0)
    grid_x = grid_x.unsqueeze(0)

    for i in range(B):
        if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
            continue

        boxes = targets[i]['boxes']  # [N, 4]
        N = len(boxes)

        # === 向量化计算面积权重 ===
        # areas: [N]
        areas = boxes[:, 2] * boxes[:, 3]
        scale_factor = 100.0
        # weights: [N] -> [N, 1, 1] 以便广播
        weights = 1.0 + 4.0 * torch.exp(-areas * scale_factor)
        weights = weights.view(N, 1, 1)

        # === 向量化计算中心点 ===
        # cx, cy: [N] -> [N, 1, 1]
        cx = (boxes[:, 0] * W).view(N, 1, 1)
        cy = (boxes[:, 1] * H).view(N, 1, 1)

        # === 核心广播计算 ===
        # (grid - center)^2 : [1, H, W] - [N, 1, 1] -> [N, H, W]
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2

        # 高斯分布: [N, H, W]
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))

        # 应用权重: [N, H, W] * [N, 1, 1]
        weighted_gaussian = gaussian * weights

        # === Max 聚合 ===
        # 对应你代码中的 torch.maximum
        # 从 [N, H, W] 压缩到 [H, W]
        if N > 0:
            # val: [H, W], idx: [H, W]
            val, _ = weighted_gaussian.max(dim=0)
            density_map[i, 0] = val

    return density_map
# def generate_density_map_gt(targets, feat_shape, device, sigma=2.0):
#     """
#     生成【面积加权】的高斯密度图 GT。
#     面积越小的物体，其高斯热图的峰值权重越高。
#     """
#     B, H, W = feat_shape
#     density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
#
#     # 坐标网格
#     y_range = torch.arange(H, device=device)
#     x_range = torch.arange(W, device=device)
#     grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
#
#     for i in range(B):
#         if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
#             continue
#
#         boxes = targets[i]['boxes']  # [N, 4] (cx, cy, w, h)
#
#         # === 核心修改：计算面积权重 ===
#         # w, h 是归一化的 (0~1)。面积 area = w * h
#         areas = boxes[:, 2] * boxes[:, 3]
#
#         # 加权公式：weight = 1 + alpha * exp(-beta * area)
#         # 这里的参数经过调优，能让极小目标权重达到 5.0，大目标维持在 1.0
#         # 你可以根据数据集微调 scale_factor (beta)
#         scale_factor = 100.0
#         weights = 1.0 + 4.0 * torch.exp(-areas * scale_factor)
#
#         # 归一化坐标转特征图绝对坐标
#         cx = boxes[:, 0] * W
#         cy = boxes[:, 1] * H
#
#         # 逐个目标生成高斯并加权
#         # (注：此处为了代码清晰使用了循环，N 通常不大，训练耗时可忽略)
#         for j in range(len(boxes)):
#             dist_sq = (grid_x - cx[j]) ** 2 + (grid_y - cy[j]) ** 2
#             gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
#
#             # 应用权重！
#             weighted_gaussian = gaussian * weights[j]
#
#             # 使用 Max 融合 (保留局部最强响应)
#             density_map[i, 0] = torch.maximum(density_map[i, 0], weighted_gaussian)
#
#     return density_map
# def generate_density_map_gt(targets, feat_shape, device, sigma=2.0):
#     """
#     targets: list[dict], 每个 dict 包含 'boxes' (N, 4) [cx, cy, w, h] 归一化坐标
#     feat_shape: (B, H, W)
#     """
#     B, H, W = feat_shape
#     density_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
#
#     # 坐标网格
#     y_range = torch.arange(H, device=device)
#     x_range = torch.arange(W, device=device)
#     grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
#
#     for i in range(B):
#         if 'boxes' not in targets[i] or len(targets[i]['boxes']) == 0:
#             continue
#
#         boxes = targets[i]['boxes']  # [N, 4]
#         # 归一化坐标转特征图绝对坐标
#         cx = boxes[:, 0] * W
#         cy = boxes[:, 1] * H
#
#         # 矢量化计算高斯
#         # shape: [N, H, W]
#         dist_sq = (grid_x.unsqueeze(0) - cx.view(-1, 1, 1)) ** 2 + \
#                   (grid_y.unsqueeze(0) - cy.view(-1, 1, 1)) ** 2
#         gaussians = torch.exp(-dist_sq / (2 * sigma ** 2))
#
#         # 取最大响应作为该像素的密度值 (适合检测任务)
#         if gaussians.shape[0] > 0:
#             val, _ = gaussians.max(dim=0)
#             density_map[i, 0] = val
#
#     return density_map

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




