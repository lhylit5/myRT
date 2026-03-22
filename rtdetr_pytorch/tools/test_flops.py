import os
import sys
import cv2
import torch
import numpy as np
from thop import profile
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.core import YAMLConfig



def main():
    # 1. 加载你的配置文件 (替换为你实际使用的 yaml 路径)
    config_file = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
    cfg = YAMLConfig(config_file)

    # 2. 实例化模型并设置为评估模式
    model = cfg.model
    model.eval()

    # 可选：如果你的模型权重中有结构重参数化（如 RepVGG），在测试前最好转为部署模式
    # if hasattr(model, 'deploy'):
    #     model.deploy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. 构造一个标准的输入张量 (RT-DETR 默认输入分辨率通常为 640x640)
    # 维度: [BatchSize, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    # 4. 使用 thop 计算 FLOPs 和 参数量
    # 注意：RT-DETR 的 forward 在推理时通常只接受 images
    macs, params = profile(model, inputs=(dummy_input,))

    # thop 输出的是 MACs (乘加操作数)，通常 1 MAC = 2 FLOPs
    # 也有很多论文直接将 MACs 作为 FLOPs 报告。这里乘以 2 转换为理论 GFLOPs
    gflops = (macs * 2) / 1e9
    params_m = params / 1e6

    print(f"=======================================")
    print(f"Model Config: {config_file}")
    print(f"Params: {params_m:.2f} M")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"=======================================")


if __name__ == '__main__':
    main()