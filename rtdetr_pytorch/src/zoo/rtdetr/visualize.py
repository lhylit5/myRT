import cv2
import numpy as np
import torchvision
from matplotlib import pyplot as plt

from .box_ops import box_cxcywh_to_xyxy

count = 0
query_count = 0
layer = 1
group = 1
bs = 1
def visualize_features(H, W, output, phase, num_channels_to_visualize=10):
    """
    可视化输出特征图的前num_channels_to_visualize个通道。

    参数:
    H (int): 特征图的高度。
    W (int): 特征图的宽度。
    output (torch.Tensor): 输出特征图，形状为[B, H*W, C]。
    num_channels_to_visualize (int): 要可视化的通道数。默认为10。
    """
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 获取批次中第一个样本的所有通道的特征图
    for e ,o in enumerate(output):
        first_sample_features = o.reshape(H, W, output.shape[2])

        # 创建一个子图网格，每个子图显示一个通道的特征图
        fig, axes = plt.subplots(1, num_channels_to_visualize, figsize=(20, 5))
        fig.suptitle(phase, fontsize=20, y=0.06)  # y=1.05 将标题向上移动，避免与子图重叠

        if num_channels_to_visualize == 1:
            axes = [axes]  # 将axes包装成列表，以便统一处理

        for i in range(num_channels_to_visualize):
            # 获取第i个通道的特征图
            channel_feature = first_sample_features[:, :, i]

            # 在子图上显示这个通道的特征图
            axes[i].set_title(f"Channel {i}")
            axes[i].imshow(channel_feature.cpu().detach().numpy(), cmap='viridis')
            axes[i].axis('off')  # 不显示坐标轴

        # plt.show()

def visualize_boxes(image, phase, bboxes, nums, color='green'):
    global count
    global layer
    global group
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    if count % 40 == 0:
    # if True:
        # image = cv2.imread(imagePath)
        # if phase != '标签':
        # 将bbox的值从[0,1]缩放到图像尺寸
        if phase == '验证':
            # bboxes = box_cxcywh_to_xyxy(bboxes)
            bboxes = bboxes.detach().cpu().numpy()
            bbox_img = bboxes
        else:
            bboxes = box_cxcywh_to_xyxy(bboxes)
            # bboxes = torchvision.ops.box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            bboxes = bboxes.detach().cpu().numpy()
            bbox_img = (bboxes * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 创建一个子图
        fig.suptitle(phase, fontsize=20, y=0.06)
        # 绘制原始图像
        ax.imshow(image)

        # 遍历nums个边界框，每个都在图上绘制
        for i, bbox in enumerate(bbox_img[:nums]):
            # 绘制边界框
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, edgecolor=color, linewidth=2))

            # 可以在这里添加文本标签，例如 i+1 表示第几个边界框
            ax.text(bbox[0], bbox[1], str(i + 1), color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

        # 移除坐标轴

        # 调整子图间距
        plt.tight_layout()
        filename = "./output/image/epoch_1/decoder/{:0>12d}_{:0>2d}_{:0>2d}.png".format(count, group, layer)

        # 保存图片
        # plt.savefig(filename)
        # plt.show()
        plt.close(fig)
    group += 1
    if group == 6:
        layer += 1
        group = 1
    if layer == 8:
        count += 1
        layer = 1

    # 显示图形
    # plt.show()

def visualize_queries(image, isTraining, bboxes, nums, queries, color='green'):
    """
    可视化query
    Args:
        image: sample图片
        phase: 训练/验证标识
        bboxes: 标签中各个真实框，xywh格式
        nums: 标签中真实框个数
        queries: queries对应的enc_topk_bboxes（根据中心位置可视化）
        color: 可视化颜色

    Returns:

    """
    global query_count, queries_center
    global layer
    global group
    global bs
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    if query_count % 40 == 0:
    # if True:
        # image = cv2.imread(imagePath)
        # if phase != '标签':
        # 将bbox的值从[0,1]缩放到图像尺寸
        if isTraining :
            # 得到标签真实框
            bboxes = box_cxcywh_to_xyxy(bboxes)
            # bboxes = torchvision.ops.box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
            bboxes = bboxes.detach().cpu().numpy()
            bbox_img = (bboxes * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
        else:
            # bboxes = box_cxcywh_to_xyxy(bboxes)
            bboxes = bboxes.detach().cpu().numpy()
            bbox_img = bboxes

        queries = queries.detach().cpu().numpy()
        queries_center = (queries * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 创建一个子图
        fig.suptitle("training" if isTraining else "test", fontsize=20, y=0.06)
        # 绘制原始图像
        ax.imshow(image)

        # 遍历nums个边界框，每个都在图上绘制
        for i, bbox in enumerate(bbox_img[:nums]):
            # 绘制边界框
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, edgecolor=color, linewidth=2))

            # 可以在这里添加文本标签，例如 i+1 表示第几个边界框
            ax.text(bbox[0], bbox[1], str(i + 1), color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

        for i, query in enumerate(queries_center):
            # 提取中心点坐标（假设query是[cx, cy, w, h]或[cx, cy]格式）
            cx, cy = query[0], query[1]

            # 绘制中心点（红色实心圆点，尺寸12）
            ax.scatter(cx, cy, color='blue', s=20, marker='o', zorder=3)

            # # 添加带背景的文本标签（偏移量防止遮挡）
            # ax.text(cx + 5, cy - 5, str(i + 1),
            #         color='white', fontsize=10,
            #         bbox=dict(facecolor='blue', alpha=0.8, boxstyle='round'))

        # 移除坐标轴

        # 调整子图间距
        plt.tight_layout()
        filename = "./output/image/epoch_1/queries/{:0>12d}_{:0>2d}.png".format(query_count, bs)

        # 保存图片
        # plt.savefig(filename)
        # plt.show()
        plt.close(fig)
    bs += 1
    # if group == 6:
    #     layer += 1
    #     group = 1
    if bs == 9:
        query_count += 1
        bs = 1

    # 显示图形
    # plt.show()