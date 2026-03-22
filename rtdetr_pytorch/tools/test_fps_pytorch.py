import torch
import time
import os
import sys
import cv2
import numpy as np
from thop import profile
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.core import YAMLConfig


def main():
    config_file = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
    weight_path = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth' # 替换为你的权重路径

    cfg = YAMLConfig(config_file)
    model = cfg.model

    # 加载权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema']['module'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.eval()
    device = torch.device('cuda:0')
    model.to(device)

    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    print("开始 Warmup (预热)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    print("开始测速...")
    iters = 200
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = iters / total_time
    ms_per_img = (total_time / iters) * 1000

    print(f"=======================================")
    print(f"Inference Time per image: {ms_per_img:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"=======================================")


if __name__ == '__main__':
    main()