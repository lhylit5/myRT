import os
import sys
import torch
import types
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. 环境设置
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core import YAMLConfig
from pycocotools.coco import COCO
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer

# =====================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 用户配置区域 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# =====================================================================

# 你想测试的模型配置文件和权重
# 建议先测 Baseline，再测 Ours，看看分布变化
# CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco_base.yml'
# CKPT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

#
CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
CKPT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'
# 数据集路径 (用于加载验证集)
COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

# 测试图片数量 (设为 -1 则跑完所有验证集)
NUM_TEST_IMAGES = 2000

# =====================================================================

# 全局累加器，用于最后求平均
GLOBAL_STATS = {
    'level_counts': [0, 0, 0],  # S3, S4, S5
    'scale_counts': [0, 0, 0],  # Small, Medium, Large
    'total_queries': 0
}


def analyze_query_stats(topk_ind, spatial_shapes, enc_outputs_coord_unact):
    """
    核心统计函数
    topk_ind: [B, 300] 选中的索引
    spatial_shapes: 特征图尺寸列表 [[80,80], [40,40], [20,20]]
    enc_outputs_coord_unact: [B, Total_Anchors, 4] 所有的 Proposals
    """
    bs, num_queries = topk_ind.shape
    device = topk_ind.device

    # 1. 计算每一层的索引范围
    limits = []
    acc = 0
    for h, w in spatial_shapes:
        acc += h * w
        limits.append(acc)
    # limits 类似 [6400, 8000, 8400]

    # 2. 统计层级分布 (Level Distribution)
    # S3: 0 ~ limits[0]
    # S4: limits[0] ~ limits[1]
    # S5: limits[1] ~ limits[2]

    mask_s3 = (topk_ind < limits[0])
    mask_s4 = (topk_ind >= limits[0]) & (topk_ind < limits[1])
    mask_s5 = (topk_ind >= limits[1])

    n_s3 = mask_s3.sum().item()
    n_s4 = mask_s4.sum().item()
    n_s5 = mask_s5.sum().item()

    GLOBAL_STATS['level_counts'][0] += n_s3
    GLOBAL_STATS['level_counts'][1] += n_s4
    GLOBAL_STATS['level_counts'][2] += n_s5

    # 3. 统计尺度分布 (Scale Distribution)
    # 获取被选中的 Query 对应的 Proposal Box
    # enc_outputs_coord_unact: [B, Total, 4]
    sel_ref_points = enc_outputs_coord_unact.gather(
        dim=1,
        index=topk_ind.unsqueeze(-1).repeat(1, 1, 4)
    )
    sel_boxes = sel_ref_points.sigmoid()  # 归一化 [cx, cy, w, h]

    # 计算相对面积 (w * h)
    # 假设图片 resize 到 640x640
    # COCO Small: < 32^2 = 1024 px^2 -> 归一化阈值 (32/640)^2 = 0.0025
    # COCO Medium: < 96^2 = 9216 px^2 -> 归一化阈值 (96/640)^2 = 0.0225

    areas = sel_boxes[..., 2] * sel_boxes[..., 3]

    thr_small = (32.0 / 640.0) ** 2
    thr_large = (96.0 / 640.0) ** 2

    n_small = (areas < thr_small).sum().item()
    n_mid = ((areas >= thr_small) & (areas < thr_large)).sum().item()
    n_large = (areas >= thr_large).sum().item()

    GLOBAL_STATS['scale_counts'][0] += n_small
    GLOBAL_STATS['scale_counts'][1] += n_mid
    GLOBAL_STATS['scale_counts'][2] += n_large

    GLOBAL_STATS['total_queries'] += (bs * num_queries)


# --- 定义 Patch 函数：带统计功能的 _get_decoder_input ---
# 这个函数会替换模型原本的方法
def patched_get_decoder_input(self, memory, spatial_shapes, denoising_class=None,
                              denoising_bbox_unact=None, samples=None, targets=None, density_map=None):
    # ==== 复制原版逻辑 (确保行为一致) ====
    bs, _, _ = memory.shape
    if self.training or self.eval_spatial_size is None:
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
    else:
        anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

    memory = valid_mask.to(memory.dtype) * memory
    output_memory = self.enc_output(memory)
    enc_outputs_class = self.enc_score_head(output_memory)
    enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

    # ---- 核心筛选逻辑 (包含你的修改) ----
    if density_map is not None:
        # 复现你在 rtdetr_decoder.py 中的逻辑
        num_s3_anchors = spatial_shapes[0][0] * spatial_shapes[0][1]
        density_score_s3 = density_map.flatten(2).permute(0, 2, 1)
        total_anchors = enc_outputs_class.shape[1]
        full_density_score = torch.zeros((bs, total_anchors, 1), device=enc_outputs_class.device,
                                         dtype=enc_outputs_class.dtype)

        if total_anchors >= num_s3_anchors:
            full_density_score[:, :num_s3_anchors, :] = density_score_s3
            enc_probs = enc_outputs_class.sigmoid()
            topk_score_cls = enc_probs.max(-1).values

            # 使用你的 alpha (这里写死为你代码里的值，或保持一致)
            alpha = 0.1
            mixed_score = topk_score_cls + alpha * full_density_score.squeeze(-1)
            _, topk_ind = torch.topk(mixed_score, self.num_queries, dim=1)
        else:
            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
    else:
        # Baseline 逻辑
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

    # ==== [插入] 统计代码 ====
    analyze_query_stats(topk_ind, spatial_shapes, enc_outputs_coord_unact)
    # ========================

    # 后续处理 (复制原版)
    reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
                                                            index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                                enc_outputs_coord_unact.shape[
                                                                                                    -1]))
    enc_topk_bboxes = F.sigmoid(reference_points_unact)

    if denoising_bbox_unact is not None:
        reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

    enc_topk_logits = enc_outputs_class.gather(dim=1, \
                                               index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

    if self.learnt_init_query:
        target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
    else:
        target = output_memory.gather(dim=1, \
                                      index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach()

    if denoising_class is not None:
        target = torch.concat([denoising_class, target], 1)

    return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


def main():
    # 1. 加载模型
    print(f"Loading model from {CONFIG_PATH}...")
    cfg = YAMLConfig(CONFIG_PATH, resume=CKPT_PATH)
    model = cfg.model
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        model.load_state_dict(state, strict=False)
    else:
        print("Warning: No checkpoint loaded!")

    model.cuda().eval()

    # 2. 【关键一步】应用 Monkey Patch
    # 将模型实例的 _get_decoder_input 方法替换为我们的带统计版本
    print("Applying Monkey Patch to inject analysis hook...")

    # === [修改前] 报错 ===
    # model.transformer._get_decoder_input = types.MethodType(patched_get_decoder_input, model.transformer)

    # === [修改后] 正确 ===
    # 这里的 self.decoder 就是 RTDETRTransformer 的实例
    if hasattr(model, 'decoder'):
        model.decoder._get_decoder_input = types.MethodType(patched_get_decoder_input, model.decoder)
    else:
        # 防御性编程：如果是其他变体，可能叫 transformer
        model.transformer._get_decoder_input = types.MethodType(patched_get_decoder_input, model.transformer)
    # 3. 准备数据
    coco = COCO(VAL_ANN)
    img_ids = coco.getImgIds()
    if NUM_TEST_IMAGES > 0:
        img_ids = img_ids[:NUM_TEST_IMAGES]

    print(f"Start analyzing {len(img_ids)} images...")

    # 4. 运行推理
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(fpath): continue

        # 简单预处理 (Resize to 640x640)
        import cv2
        img = cv2.imread(fpath)
        img = cv2.resize(img, (640, 640))
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

        with torch.no_grad():
            model(tensor)

    # 5. 输出汇总报告
    total = GLOBAL_STATS['total_queries']
    if total == 0: return

    l_cnt = GLOBAL_STATS['level_counts']
    s_cnt = GLOBAL_STATS['scale_counts']

    print("\n" + "=" * 50)
    print(f" Query Analysis Report (Total Queries: {total})")
    print("=" * 50)

    print(f"[Level Distribution] 来自哪个特征层?")
    print(f"  S3 (Stride  8, Small ): {l_cnt[0]:6d} ({l_cnt[0] / total * 100:.2f}%)")
    print(f"  S4 (Stride 16, Medium): {l_cnt[1]:6d} ({l_cnt[1] / total * 100:.2f}%)")
    print(f"  S5 (Stride 32, Large ): {l_cnt[2]:6d} ({l_cnt[2] / total * 100:.2f}%)")

    print(f"\n[Scale Distribution] 预测框有多大? (Based on COCO definitions)")
    print(f"  Small  (Area < 32^2) : {s_cnt[0]:6d} ({s_cnt[0] / total * 100:.2f}%)")
    print(f"  Medium (32^2~96^2)   : {s_cnt[1]:6d} ({s_cnt[1] / total * 100:.2f}%)")
    print(f"  Large  (Area > 96^2) : {s_cnt[2]:6d} ({s_cnt[2] / total * 100:.2f}%)")
    print("=" * 50)
    print("注：如果 S3 占比很高，但 Small 占比很低，说明 S3 层对应的框回归得很大（特征错配）。")


if __name__ == "__main__":
    main()