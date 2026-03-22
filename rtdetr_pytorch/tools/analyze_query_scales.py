#可视化不同尺寸特征图目标大小分布
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from tqdm import tqdm

# 路径补丁，确保能导入 src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.core import YAMLConfig

# ================= 配置区域 =================
# CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco_se.yml'
# CKPT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/small_obj/checkpoint0071.pth'

CONFIG_PATH = '../configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
CKPT_PATH = '../tools/output/rtdetr_r50vd_6x_coco/base/checkpoint0071.pth'

COCO_ROOT = '../configs/dataset/coco'
VAL_IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations/instances_val2017.json')

SCORE_THRESH = 0.3  # 置信度大于 0.4 视为有效检出
MAX_IMAGES = 1000  # 跑 1000 张图足够画出平滑的统计分布图了
INPUT_SIZE = 640


# ===========================================

def load_model(config_path, ckpt_path):
    print(f"Loading model from {ckpt_path}...")
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    model = cfg.model
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = state_dict.get('ema', {}).get('module', state_dict.get('model', state_dict))
        model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def main():
    model = load_model(CONFIG_PATH, CKPT_PATH)
    coco = COCO(VAL_ANN_FILE)
    img_ids = coco.getImgIds()[:MAX_IMAGES]

    # 用于存储不同特征图层级检测到的目标面积 (原图尺度)
    areas_s3 = []
    areas_s4 = []
    areas_s5 = []

    print("开始输入真实图像，提取 Query 尺度分布...")
    for img_id in tqdm(img_ids):
        # 1. 严格使用真实图像进行预处理
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        orig_img = cv2.imread(img_path)
        # 按照 RT-DETR 标准预处理
        img_resized = cv2.resize(orig_img, (INPUT_SIZE, INPUT_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # 2. 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)

        if 'topk_indexes' not in outputs:
            raise KeyError("模型输出中没有 'topk_indexes'！请检查 decoder 是否正确修改。")

        # 取出预测结果
        scores = outputs['pred_logits'].sigmoid().max(-1).values[0]  # [num_queries]
        boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
        indices = outputs['topk_indexes'][0]  # [num_queries]

        # 3. 过滤出高置信度的真实检出
        valid_mask = scores > SCORE_THRESH
        valid_boxes = boxes[valid_mask]
        valid_indices = indices[valid_mask]

        # 4. 统计面积与索引来源
        for box, idx in zip(valid_boxes, valid_indices):
            # 将归一化宽高还原为 640x640 尺度的面积
            w = box[2].item() * INPUT_SIZE
            h = box[3].item() * INPUT_SIZE
            area = w * h

            idx_val = idx.item()
            # 索引划分规则：S3 (0~6399), S4 (6400~7999), S5 (8000~8399)
            if idx_val < 6400:
                areas_s3.append(area)
            elif idx_val < 8000:
                areas_s4.append(area)
            else:
                areas_s5.append(area)

    print(f"\n统计完成！真实有效 Query 数量分布：")
    print(f"S3 (Stride 8, SOEM Enhanced): {len(areas_s3)} 个")
    print(f"S4 (Stride 16):               {len(areas_s4)} 个")
    print(f"S5 (Stride 32):               {len(areas_s5)} 个")

    if len(areas_s3) == 0 and len(areas_s4) == 0:
        print("警告：依然只有 S5 有数据，请检查 decoder 中 topk_ind 的拼接逻辑是否有误。")
        return

    # ================= 绘图部分 =================
    plt.figure(figsize=(10, 6))

    # 面积跨度极大，计算等效边长 (平方根) 方便可视化
    data_s3 = np.sqrt(areas_s3) if len(areas_s3) > 0 else []
    data_s4 = np.sqrt(areas_s4) if len(areas_s4) > 0 else []
    data_s5 = np.sqrt(areas_s5) if len(areas_s5) > 0 else []

    # 绘制核密度估计图
    if len(data_s3) > 0:
        sns.kdeplot(data_s3, fill=True, label='S3 (Stride 8, Enhanced)', color='#1f77b4', alpha=0.6, linewidth=2)
    if len(data_s4) > 0:
        sns.kdeplot(data_s4, fill=True, label='S4 (Stride 16)', color='#ff7f0e', alpha=0.5)
    if len(data_s5) > 0:
        sns.kdeplot(data_s5, fill=True, label='S5 (Stride 32)', color='#2ca02c', alpha=0.5)

    plt.title('Distribution of Target Sizes Selected by Different Feature Maps', fontsize=14, fontweight='bold')
    plt.xlabel('Object Equivalent Side Length ( $\sqrt{Area}$ in pixels )', fontsize=12)
    plt.ylabel('Density of Selected Queries', fontsize=12)

    # 划定小目标阈值线 (32像素)
    plt.axvline(x=32, color='red', linestyle='--', linewidth=2, label='Small Object Threshold (32 px)')

    # 将 X 轴限制在 0~300 像素边长，否则大目标会把图表拉得太长，导致小目标峰值看不清
    plt.xlim(0, 300)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = 'query_size_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 完美！可视化图表已保存至: {save_path}")


if __name__ == '__main__':
    main()