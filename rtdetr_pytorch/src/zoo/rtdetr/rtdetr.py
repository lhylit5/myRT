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

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
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
        for i, target in enumerate(targets):
            # imageId = int(target['image_id'])
            # imagePath = 'D:/pythonProject/RT-DETR-main/rtdetr_pytorch/configs/dataset/coco/val2017/{:012}.jpg'.format(imageId)
            # image = cv2.imread(imagePath)
            boxes = target['boxes']
            # visualize_boxes(samples[i], 'train', boxes, boxes.shape[0])
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, samples, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
