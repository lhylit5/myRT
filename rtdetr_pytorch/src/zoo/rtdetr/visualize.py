import cv2
import numpy as np
import torchvision
from matplotlib import pyplot as plt


def visualize_features(H, W, output, phase, num_channels_to_visualize=10):
    """
    可视化输出特征图的前num_channels_to_visualize个通道。

    参数:
    H (int): 特征图的高度。
    W (int): 特征图的宽度。
    output (torch.Tensor): 输出特征图，形状为[B, H*W, C]。
    num_channels_to_visualize (int): 要可视化的通道数。默认为10。
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 获取批次中第一个样本的所有通道的特征图
    first_sample_features = output[0].reshape(H, W, output.shape[2])

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

    plt.show()

def visualize_boxes(imagePath, bboxes, nums):
    # bbox = bboxes
    # bbox = torchvision.ops.box_convert(bbox, in_fmt='cxcywh', out_fmt='xyxy')
    # bbox = bbox.detach().cpu().numpy()
    #
    # # 将bbox的值从[0,1]缩放到图像尺寸
    # bbox_img = (bbox * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
    # bbox_img = bbox_img.astype(int)
    # for bbox in bbox_img[:nums]:
    #     # 确保每个 bbox 是整数
    #     bbox = bbox.astype(int)
    #
    #     # 绘制每个矩形
    #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
    #
    # # 显示图像
    # cv2.imshow('Bboxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = cv2.imread(imagePath)

    # 将bbox的值从[0,1]缩放到图像尺寸
    bboxes = torchvision.ops.box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')
    bboxes = bboxes.detach().cpu().numpy()
    bbox_img = (bboxes * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)

    # 创建一个图形，每个边界框一个子图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 创建一个子图
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 绘制原始图像
    ax.imshow(image_rgb)

    # 遍历nums个边界框，每个都在图上绘制
    for i, bbox in enumerate(bbox_img[:nums]):
        # 绘制边界框
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                   fill=False, edgecolor='green', linewidth=1))

        # 可以在这里添加文本标签，例如 i+1 表示第几个边界框
        ax.text(bbox[0], bbox[1], str(i + 1), color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    # 移除坐标轴
    ax.axis('off')

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()
