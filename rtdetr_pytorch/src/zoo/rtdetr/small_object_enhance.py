# src/zoo/rtdetr/small_enhance.py
# 小目标信息增强（基于 DQ-DETR 风格）
# SpatialGate 使用 Conv_BN（可选 BN）
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.core import register
except Exception:
    def register(cls): return cls

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
# ----------------------------
class ConvSimple(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, relu=True, bias=False):
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
# 参数 use_bn 控制是否启用 BN（默认 True）
# ----------------------------
class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7, use_bn=True):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.compress = ChannelPool()
        # 使用 Conv_BN: 当 out_channel=1 且 bn=True 时会创建 BatchNorm2d(1)，这是允许的
        # 若 batch size 很小（如 1），BN 可能不稳定，因此提供 use_bn 开关
        self.spatial = Conv_BN(2, 1, kernel_size=kernel_size, stride=1, padding=pad, relu=False, bn=use_bn)

    def forward(self, x):
        x_comp = self.compress(x)    # [B,2,H,W]
        x_out = self.spatial(x_comp) # [B,1,H,W]
        return torch.sigmoid(x_out)


# ----------------------------
# ChannelGate: DQ-DETR 风格 MLP（不含归一化）
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
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                      stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                tensor_flatten = x.view(x.size(0), x.size(1), -1)
                s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
                lse_pool = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
                channel_att_raw = self.mlp(lse_pool)
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
    def __init__(self, in_ch=256, mid_ch=256, ccm_cfg=(256, 256), dilation=2,
                 use_aux=False, reduction_ratio=16, pool_types=['avg', 'max'], use_bn=True):
        """
        in_ch: 输入特征通道（通常 256）
        mid_ch: 1x1 映射通道
        ccm_cfg: CCM 每层通道配置（list/tuple），最后一项为 Fc 的通道数
        dilation: 膨胀率
        use_aux: 保留接口
        use_bn: SpatialGate 中是否使用 BatchNorm（默认 True，参照 DQ-DETR）
        """
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.use_aux = use_aux

        # 1x1 映射（不含归一化）
        self.conv1 = ConvSimple(in_ch, mid_ch, kernel_size=1, padding=0, relu=True, bias=False)

        # CCM（膨胀卷积序列），不使用归一化（保持轻量）
        layers = []
        in_c = mid_ch
        for out_c in ccm_cfg:
            layers.append(ConvSimple(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation, relu=True, bias=False))
            in_c = out_c
        self.ccm = nn.Sequential(*layers)
        self.ccm_out_ch = ccm_cfg[-1]

        # 空间与通道门（SpatialGate 使用 Conv_BN）
        self.spatial_gate = SpatialGate(kernel_size=7, use_bn=use_bn)
        self.channel_gate = ChannelGate(self.ccm_out_ch, reduction_ratio=reduction_ratio, pool_types=pool_types)

        # 可学习系数
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            raise ValueError("feats must be list/tuple")

        S = feats[0]
        B, C0, H, W = S.shape

        # 1x1 -> mid_ch
        v = self.conv1(S)  # [B, mid_ch, H, W]

        # CCM -> Fc
        Fc = self.ccm(v)   # [B, ccm_out_ch, H, W]

        # 空间注意力
        Ws = self.spatial_gate(Fc)  # [B,1,H,W]
        mask_for_S = F.interpolate(Ws, size=(H, W), mode='bilinear', align_corners=False)
        enhanced_S = S * (1.0 + self.alpha * mask_for_S)

        # 通道注意力
        Wc = self.channel_gate(Fc)  # [B, ccm_out_ch, 1, 1]
        if Wc.size(1) == C0:
            enhanced_S = enhanced_S * Wc
        else:
            # 保守方案：用 Wc 的均值作为整体缩放（避免新增参数）
            Wc_scalar = Wc.mean(dim=1, keepdim=True)  # [B,1,1,1]
            enhanced_S = enhanced_S * Wc_scalar

        out_feats = list(feats)
        out_feats[0] = enhanced_S
        return out_feats
