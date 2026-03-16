import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_grid(ax, grid_size=7):
    # 绘制背景网格
    for i in range(grid_size + 1):
        ax.plot([0, grid_size], [i, i], color='lightgray', linewidth=1.5)
        ax.plot([i, i], [0, grid_size], color='lightgray', linewidth=1.5)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_conv(ax, center, dilation=1, grid_size=7):
    draw_grid(ax, grid_size)
    cx, cy = center

    # 蓝色采样点，红色中心点
    color_point = '#4dabf7'
    color_center = '#fa5252'

    # 计算采样点的坐标偏移 (-1, 0, 1) 乘以 dilation
    offsets = [-1, 0, 1]

    # 绘制采样点
    min_x, max_x = cx, cx
    min_y, max_y = cy, cy

    for dx in offsets:
        for dy in offsets:
            px = cx + dx * dilation
            py = cy + dy * dilation

            # 更新感受野边界
            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)

            # 区分中心点和其他点
            color = color_center if dx == 0 and dy == 0 else color_point
            rect = patches.Rectangle((px, grid_size - 1 - py), 1, 1,
                                     linewidth=1.5, edgecolor='#1971c2', facecolor=color, alpha=0.8)
            ax.add_patch(rect)

    # 绘制感受野虚线框
    rf_rect = patches.Rectangle((min_x - 0.1, grid_size - 1 - max_y - 0.1),
                                (max_x - min_x + 1.2), (max_y - min_y + 1.2),
                                fill=False, edgecolor='#e64980', linestyle='--', linewidth=2.5)
    ax.add_patch(rf_rect)


# 设置画图参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 图1：标准卷积
draw_conv(ax1, center=(3, 3), dilation=1)
ax1.set_title('(a)标准卷积 (Dilation=1)\n感受野:3x3 ', fontsize=14, pad=15)

# 图2：空洞卷积
draw_conv(ax2, center=(3, 3), dilation=2)
ax2.set_title('(b)空洞卷积 (Dilation=2)\n感受野:5x5 ', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig('dilated_conv_comparison.pdf', dpi=300, bbox_inches='tight')  # 保存为矢量图 PDF
plt.savefig('dilated_conv_comparison.png', dpi=300, bbox_inches='tight')  # 同时保存 PNG
print("图片已保存至当前目录！")