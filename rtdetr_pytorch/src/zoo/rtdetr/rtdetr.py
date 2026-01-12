"""by lyuwenyu
"""
import cv2
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np
from matplotlib import pyplot as plt

from src.core import register


__all__ = ['RTDETR', ]

from .visualize import visualize_boxes


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, use_density_query_selection=False):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        # 保存开关状态
        self.use_density_query_selection = use_density_query_selection
        
    def forward(self, x, targets=None):
        samples = []
        for i, temp in enumerate(x):
            temp = temp.permute(1, 2, 0).cpu().numpy()
            temp = np.clip(temp, 0, 1)
            samples.append(temp)
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # samples = []
        # for i, temp in enumerate(x):
        #     temp = temp.permute(1, 2, 0).cpu().numpy()
        #     temp = np.clip(temp, 0, 1)
        #     samples.append(temp)

        if targets is not None:
            for i, target in enumerate(targets):
                # imageId = int(target['image_id'])
                # imagePath = 'D:/pythonProject/RT-DETR-main/rtdetr_pytorch/configs/dataset/coco/val2017/{:012}.jpg'.format(imageId)
                # image = cv2.imread(imagePath)
                boxes = target['boxes']
                # visualize_boxes(samples[i], 'train', boxes, boxes.shape[0])
        x = self.backbone(x)
        # === 修改 1: 接收 4 个返回值 ===
        # 原来是: x, pred_density_map = self.encoder(x)
        x, pred_density_map, feat_before, feat_after = self.encoder(x)
        # 2. 将密度图传给 decoder (仅当开关开启且有密度图时)
        # 注意：你需要修改 Decoder 的 forward 签名来接收这个参数
        density_map_for_decoder = pred_density_map if self.use_density_query_selection else None
        x = self.decoder(x, samples, targets, density_map=density_map_for_decoder)
        # === 【新增】将密度图加入输出字典 ===
        # 只有在训练阶段且密度图存在时才需要加入
        if pred_density_map is not None:
            x['pred_density_map'] = pred_density_map
            # 仅在推理/验证模式下保存特征图，节省训练显存
        if not self.training:
            if feat_before is not None:
                x['feat_before'] = feat_before
            if feat_after is not None:
                x['feat_after'] = feat_after
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
