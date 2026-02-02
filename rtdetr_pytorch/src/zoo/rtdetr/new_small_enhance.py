# src/zoo/rtdetr/small_enhance.py
# 小目标信息增强（基于 DQ-DETR 风格）
# 优化点：调整通道数减少计算量；修复 ConvSimple 缺失 Bias 问题
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.core import register
except Exception:
    def register(cls):
        return cls

__all__ = ['SmallObjectEnhance']


# ----------------------------
# Conv + BN + ReLU（DQ-DETR 风格）
# ----------------------------
class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ----------------------------
# 基础 Conv（不含归一化），用于 CCM 等
# 修正：默认 bias=True，因为没有 BN
# ----------------------------
class ConvSimple(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, relu=True, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ----------------------------
# ChannelPool (max + mean)
# ----------------------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


# ----------------------------
# SpatialGate：compress -> Conv_BN -> sigmoid
# ----------------------------
class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7, use_bn=True):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.compress = ChannelPool()
        # 注意：这里 Conv_BN 的 bias 设为 False 是可以的，因为后面接了 BN (如果 use_bn=True)
        # 如果 use_bn=False，Conv_BN 内部逻辑需要处理 bias。
        # 为保险起见，我们让 Conv_BN 内部处理：如果 bn=False，建议 bias=True。
        # 这里为了简单，保持你的原逻辑，但在 SpatialGate 这种attention生成中，bias影响不大。
        self.spatial = Conv_BN(2, 1, kernel_size=kernel_size, stride=1, padding=pad, relu=False, bn=use_bn,
                               bias=not use_bn)

    def forward(self, x):
        x_comp = self.compress(x)  # [B,2,H,W]
        x_out = self.spatial(x_comp)  # [B,1,H,W]
        return torch.sigmoid(x_out)


# ----------------------------
# ChannelGate: MLP
# ----------------------------
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        inner = max(1, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, inner),
            nn.ReLU(inplace=True),
            nn.Linear(inner, gate_channels)
        )
        self.pool_types = list(pool_types)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # 显式指定 dim，防止 warning
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            else:
                continue

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


# ----------------------------
# SmallObjectEnhance（主类）
# ----------------------------
@register
class SmallObjectEnhance(nn.Module):
    def __init__(self,
                 in_ch=256,
                 mid_ch=256,  # [优化] 默认为 256，原 512 太重
                 ccm_cfg=(256, 256, 256, 256),  # [优化] 默认为 256，保持与 RT-DETR hidden_dim 一致
                 dilation=2,
                 use_aux=False,
                 reduction_ratio=16,
                 pool_types=['avg', 'max'],
                 use_bn=True,
                 use_feature_enhance=True):
        """
        in_ch: 输入特征通道（通常 256）
        mid_ch: 1x1 映射通道。建议设为 256 以保持轻量。
        ccm_cfg: CCM 序列配置。
        """
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.use_aux = use_aux
        self.use_feature_enhance = use_feature_enhance

        # 1x1 映射（不含归一化，bias=True）
        self.conv1 = ConvSimple(in_ch, mid_ch, kernel_size=1, padding=0, relu=True, bias=True)

        # CCM（膨胀卷积序列）
        # [重要] 移除了 BN，所以 bias 必须为 True，否则卷积无法学习偏移量
        layers = []
        in_c = mid_ch
        for out_c in ccm_cfg:
            layers.append(
                ConvSimple(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation, relu=True, bias=True))
            in_c = out_c
        self.ccm = nn.Sequential(*layers)
        self.ccm_out_ch = ccm_cfg[-1]

        # === 密度图预测头 ===
        # 输出 1 通道密度图
        self.density_head = nn.Conv2d(self.ccm_out_ch, 1, kernel_size=1, bias=True)  # 建议 bias=True
        self.relu = nn.ReLU(inplace=True)

        if self.use_feature_enhance:
            # 空间与通道门
            self.spatial_gate = SpatialGate(kernel_size=7, use_bn=use_bn)
            self.channel_gate = ChannelGate(self.ccm_out_ch, reduction_ratio=reduction_ratio, pool_types=pool_types)

            # [优化] 初始化为 0，让训练初期保持恒等映射，避免干扰 Backbone 特征
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        # RT-DETR 传入的是 list[Tensor]
        if not isinstance(feats, (list, tuple)):
            raise ValueError("feats must be list/tuple")

        S = feats[0]  # S3 feature, [B, 256, H, W]
        B, C0, H, W = S.shape

        # 1. 生成上下文特征 Fc
        v = self.conv1(S)  # [B, mid_ch, H, W]
        Fc = self.ccm(v)  # [B, ccm_out_ch, H, W]

        # 2. 生成预测密度图 (用于 Loss 计算)
        pred_density_map = self.relu(self.density_head(Fc))  # [B, 1, H, W]

        # 3. 特征增强
        enhanced_S = S
        if self.use_feature_enhance:
            # 空间注意力 (Spatial Attention)
            Ws = self.spatial_gate(Fc)  # [B,1,H,W]
            # 上采样 Mask (通常 CCM 不改变分辨率，但以防万一)
            if Ws.shape[2:] != (H, W):
                mask_for_S = F.interpolate(Ws, size=(H, W), mode='bilinear', align_corners=False)
            else:
                mask_for_S = Ws

            # 残差增强: S * (1 + alpha * mask)
            enhanced_S = S * (1.0 + self.alpha * mask_for_S)

            # 通道注意力 (Channel Attention)
            # 假设 Fc 和 S 通道数一致 (均为 256)，则可以直接加权
            # 如果不一致 (mid_ch=512)，Wc 维度是 512，无法直接乘 S(256)
            Wc = self.channel_gate(Fc)  # [B, ccm_out_ch, 1, 1]

            if Wc.size(1) == C0:
                enhanced_S = enhanced_S * Wc
            else:
                # 维度不匹配时的 fallback：取均值作为全局 scalar 缩放
                # 这是一个保护措施，防止 mid_ch != in_ch 时报错
                enhanced_S = enhanced_S * Wc.mean(dim=1, keepdim=True)

        # 替换原 S3 特征
        out_feats = list(feats)
        out_feats[0] = enhanced_S

        return out_feats, pred_density_map