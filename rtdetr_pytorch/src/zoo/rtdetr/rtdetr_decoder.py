"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .visualize import visualize_boxes, visualize_queries

from src.core import register

from pathlib import Path


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 【精妙之处】：截取 FFN 前的特征
        pre_ffn_feat = tgt
        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt, pre_ffn_feat

class O2M_FFN(nn.Module):
    """
    专为 O2M 辅助分支设计的独立 FFN。
    结构与主干 TransformerDecoderLayer 内部的 FFN 完全一致，保证特征分布的稳定性。
    """
    def __init__(self, d_model=256, dim_feedforward=1024, dropout=0., activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        # 1. 升维映射 + 激活
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        # 2. 残差连接
        tgt = tgt + self.dropout2(tgt2)
        # 3. 层归一化
        tgt = self.norm(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                samples=None,
                dn_meta=None,
                ffn_o2m=None,
                score_head_o2m=None):  # === [新增] 接收独立的 O2M 回归头 ===
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_logits_o2m = []
        dec_out_bboxes_o2m = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            # --- 1. 共享的 Attention 计算，接收两个特征 ---
            output, pre_ffn_feat = layer(output, ref_points_input, memory, memory_spatial_shapes,
                                         memory_level_start_index, attn_mask, memory_mask, query_pos_embed)

            # --- 2. 主分支 (O2O) 预测 ---
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

                # --- 3. 辅助分支 (O2M) 解耦预测 ---
                if ffn_o2m is not None:
                    # (a) 特征解耦：使用 pre_ffn_feat 过独立的 O2M FFN，进入多目标特征空间
                    o2m_feat = ffn_o2m[i](pre_ffn_feat)

                    # (b) 共享预测头：复用主干的分类和回归头，由于特征空间已解耦，不再产生撕扯
                    # dec_out_logits_o2m.append(score_head[i](o2m_feat))
                    dec_out_logits_o2m.append(score_head_o2m[i](o2m_feat))

                    if i == 0:
                        # O2M 的回归同样基于主干解耦出的 ref_points_detach
                        inter_ref_bbox_o2m = F.sigmoid(bbox_head[i](o2m_feat) + inverse_sigmoid(ref_points_detach))
                        dec_out_bboxes_o2m.append(inter_ref_bbox_o2m)
                    else:
                        # 重点：O2M 的偏移基准必须是上一层主分支的 ref_points
                        inter_ref_bbox_o2m = F.sigmoid(bbox_head[i](o2m_feat) + inverse_sigmoid(ref_points))
                        dec_out_bboxes_o2m.append(inter_ref_bbox_o2m)

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            # --- 4. 【铁律】Reference points 只由主分支 (O2O) 更新 ---
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        if self.training and len(dec_out_logits_o2m) > 0:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(
                dec_out_bboxes_o2m), torch.stack(dec_out_logits_o2m)

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), None, None


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 use_density_query_selection=False,
                 use_o2m=False,
                 density_weight_init=None):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        self.use_density_query_selection = use_density_query_selection
        self.use_o2m = use_o2m

        # === [新增] 定义层级密度权重为可学习参数 ===
        # 初始化为 [1.0, 0.1, 0.1]，对应 S3, S4, S5
        # 这样网络初始状态会偏向 S3，但后续可以自动调整 S4/S5 的重要性

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # === [新增/修改] 独立实例化 O2M 分类头和回归头 ===
        if self.use_o2m:
            # 采用 1024 维度的标准 FFN，带残差和 LayerNorm
            self.dec_ffn_o2m = nn.ModuleList([
                O2M_FFN(hidden_dim, dim_feedforward, dropout, activation)
                for _ in range(num_decoder_layers)
            ])
            self.dec_score_head_o2m = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes)
                for _ in range(num_decoder_layers)
            ])
        else:
            self.dec_ffn_o2m = None
            self.dec_score_head_o2m = None

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # === [新增] 初始化 O2M FFN 的参数 ===
        if self.use_o2m and self.dec_ffn_o2m is not None:
            for ffn_ in self.dec_ffn_o2m:
                # 使用 Xavier 初始化保证初始梯度的稳定
                init.xavier_uniform_(ffn_.linear1.weight)
                init.constant_(ffn_.linear1.bias, 0)
                init.xavier_uniform_(ffn_.linear2.weight)
                init.constant_(ffn_.linear2.bias, 0)
        if self.use_o2m and self.dec_score_head_o2m is not None:
            for cls_ in self.dec_score_head_o2m:
                init.constant_(cls_.bias, bias)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
                                                                                                                                                                                                                                                                        
        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           samples=None,
                           targets=None,
                           density_map=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        # ====================================================================================
        # === Point 2 核心逻辑：基于密度与门控的 S3 Query 智能筛选 ===
        # ====================================================================================
        # 2. 准备 S3 密度加成
        if self.use_density_query_selection and density_map is not None:
            # 1. 原始分类分数 (Sigmoid)
            enc_probs = enc_outputs_class.sigmoid()
            topk_score_cls = enc_probs.max(-1).values  # [B, Total_Anchors]
            # === 参数设置 ===
            num_queries = self.num_queries  # 300
            num_safe = 200  # 【保底名额】：留给高分大物体，雷打不动
            # 如果你想要更激进的 rescue，可以调小这个值，比如 100

            # === Step 1: VIP 保底筛选 (仅基于 Cls) ===
            # 选出最自信的 Top-150，这些通常是大物体或极清晰的小物体
            # topk_inds_safe: [B, 150]
            scores_safe, topk_inds_safe = torch.topk(topk_score_cls, num_safe, dim=1)

            # === Step 2: 准备互补评分 (针对剩余名额) ===
            # 获取密度图并展平
            density_map = density_map.sigmoid()
            s3_h, s3_w = spatial_shapes[0]
            num_s3 = s3_h * s3_w

            # 构造全尺寸密度分数 [B, Total]
            # 默认给负分，防止 S4/S5 被错误加分
            full_density_score = torch.zeros_like(topk_score_cls)

            # 填充 S3 部分
            density_s3 = density_map.flatten(2).squeeze(1)  # [B, S3]
            if valid_mask is not None:
                density_s3 = density_s3 * valid_mask[:, :num_s3, 0]
            full_density_score[:, :num_s3] = density_s3

            # 【核心公式】：互补评分
            # Score_new = Cls + alpha * Density * (1 - Cls)^beta
            # 这个公式本身就有“不公平”属性：它专门偏袒“分类低但密度高”的点
            alpha = 1.0
            beta = 1.0
            uncertainty = torch.pow(1.0 - topk_score_cls, beta)
            boost_score = alpha * full_density_score * uncertainty
            mixed_score = topk_score_cls + boost_score

            # === Step 3: 排除已入选 VIP 的点 ===
            # 我们要在 mixed_score 中选剩下的 150 个，但不能选已经在 Step 1 选过的
            # 方法：将已选位置的分数设为 -inf

            # scatter 的 dim=1
            # src 必须是 tensor，这里用 -inf 填充
            inf_tensor = torch.full_like(scores_safe, float('-inf'))

            # 克隆一份用于修改
            mask_mixed_score = mixed_score.clone()
            # 将 topk_inds_safe 对应位置的分数抹除
            mask_mixed_score.scatter_(1, topk_inds_safe, inf_tensor)

            # === Step 4: 互补竞技场筛选 (选剩余的 Top-150) ===
            num_rescue = num_queries - num_safe
            _, topk_inds_rescue = torch.topk(mask_mixed_score, num_rescue, dim=1)

            # === Step 5: 合并 ===
            # cat: [B, 300]
            topk_ind = torch.cat([topk_inds_safe, topk_inds_rescue], dim=1)

            # (可选) 重新按照分数排个序，虽然 DETR 不强制要求 Query 有序，但看着舒服
            # gather 对应的分数并排序... (一般不需要)

        else:
            # 原始逻辑
            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
       

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        # top300 query的cx,cy,w,h
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if targets is not None:
            for i, target in enumerate(targets):
                boxes = target['boxes']
                visualize_queries(samples[i], self.training, boxes, boxes.shape[0], enc_topk_bboxes[i])
        # for i in range(1):
        #     boxes = enc_topk_bboxes[i]
        #     chunks = torch.chunk(boxes, n, dim=0)
        #     for j, chunk in enumerate(chunks):
        #         visualize_boxes(samples[i], f'After Encoder,{j+1}group', chunk, chunk.shape[0])
            # visualize_boxes(samples[i], f'After Encoder,{i + 1}group', boxes, 30)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits, topk_ind

    def forward(self, feats, samples, targets=None, density_map=None):
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries,
                                                         self.denoising_class_embed, num_denoising=self.num_denoising,
                                                         label_noise_ratio=self.label_noise_ratio,
                                                         box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact, samples, targets,
                                    density_map)

        # === [修改] 仅传入额外的 O2M 解耦 FFN，不传 Head ===
        ffn_o2m_input = self.dec_ffn_o2m if (self.training and self.use_o2m) else None
        score_head_o2m_input = self.dec_score_head_o2m if (self.training and self.use_o2m) else None

        out_bboxes, out_logits, out_bboxes_o2m, out_logits_o2m = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,  # 共享
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            samples=samples,
            dn_meta=dn_meta,
            ffn_o2m=ffn_o2m_input,
            score_head_o2m=score_head_o2m_input)  # 仅传入特征解耦层

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            if out_logits_o2m is not None and out_bboxes_o2m is not None:
                _, out_logits_o2m = torch.split(out_logits_o2m, dn_meta['dn_num_split'], dim=2)
                _, out_bboxes_o2m = torch.split(out_bboxes_o2m, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'topk_indexes': topk_ind}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        # === 封装 O2M 输出 ===
        if self.training and out_logits_o2m is not None and out_bboxes_o2m is not None and self.use_o2m:
            out['o2m_outputs'] = {
                'pred_logits': out_logits_o2m[-1],
                'pred_boxes': out_bboxes_o2m[-1]
            }
            if self.aux_loss:
                out['o2m_outputs']['aux_outputs'] = self._set_aux_loss(
                    out_logits_o2m[:-1],
                    out_bboxes_o2m[:-1]
                )

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
