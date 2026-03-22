import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.core import register
except Exception:
    def register(cls):
        return cls

__all__ = ['SmallObjectEnhance']


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 1. X方向和Y方向的自适应池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # 2. 共享的 1x1 卷积变换
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # 3. 分别恢复通道
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # X轴和Y轴分解
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接处理 (利用通道相关性)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 双向加权
        out = a_h * a_w
        return out



class DSAM(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation_list=[1, 2, 4]):
        super().__init__()
        self.mid_channel = mid_channel

        # 1x1 降维/映射
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )

        # 并行多分支 (Multi-Branch)
        self.branches = nn.ModuleList()
        for d in dilation_list:
            self.branches.append(nn.Sequential(
                nn.Conv2d(mid_channel, mid_channel, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True)
            ))

        # 注意力融合层 (Selection Mechanism)
        # 输入: GlobalAvgPool(Sum(Branches)) -> FC -> Softmax -> Weights
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(mid_channel, max(mid_channel // 16, 32), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(mid_channel // 16, 32), len(dilation_list) * mid_channel, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        # 输出层
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),  # 这里加了BN更稳，也可以去掉保持你原来的风格
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_mid = self.conv_in(x)

        # 1. 计算各分支特征
        feats = [branch(x_mid) for branch in self.branches]  # List of [B, C, H, W]
        feats_stack = torch.stack(feats, dim=1)  # [B, 3, C, H, W]

        # 2. 融合引导 (SKNet 思想: U = Sum(Branches))
        U = torch.sum(feats_stack, dim=1)  # [B, C, H, W]

        # 3. 生成通道级选择权重
        S = self.avg_pool(U).view(b, -1)  # [B, C]
        Z = self.fc(S)  # [B, 3*C]

        # Reshape to [B, 3, C]
        weights = Z.view(b, len(self.branches), self.mid_channel)
        weights = self.softmax(weights).unsqueeze(-1).unsqueeze(-1)  # [B, 3, C, 1, 1]

        V = torch.sum(feats_stack * weights, dim=1)  # [B, C, H, W]

        return self.conv_out(V)


@register
class SmallObjectEnhance(nn.Module):
    def __init__(self, in_ch=256, mid_ch=256, out_ch=256, use_feature_enhance=True):
        super().__init__()
        self.use_feature_enhance = use_feature_enhance


        self.dsam = DSAM(in_ch, mid_ch, out_ch, dilation_list=[1, 2, 4])
        if self.use_feature_enhance:
            self.coord_att = CoordAtt(out_ch, out_ch)

        # 密度图预测头
        self.density_head = nn.Conv2d(out_ch, 1, 1)
        # self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        # 可学习系数 (保持你的风格，初始化为 1.0)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            raise ValueError("feats must be list/tuple")

        S = feats[0]  # S3 Feature [B, 256, H, W]

        Fc = self.dsam(S)  # [B, 256, H, W]

        # 2. 生成密度图 (用于 Loss)
        pred_density_map = self.density_head(Fc)

        # 3. 特征增强 (Coordinate Attention)
        enhanced_S = S
        if self.use_feature_enhance:
            att_mask = self.coord_att(Fc)
            enhanced_S = S * (1.0 + self.alpha * att_mask)

        out_feats = list(feats)
        out_feats[0] = enhanced_S
        return out_feats, pred_density_map