import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle


def draw_grid_block_svg():
    # 参数
    nx, ny = 10, 8  # 网格数量 (宽x深)
    step = 1  # 单个格子大小
    width = nx * step
    height = ny * step
    depth = 2.0  # 整体厚度

    # 颜色
    top_color = '#C1E1C1'  # 浅绿
    side_color = '#8FBC8F'  # 深绿
    edge_color = '#5F9EA0'  # 边框线

    fig, ax = plt.subplots(figsize=(8, 6))

    # 投影参数
    angle = 45
    scale = 0.5
    theta = np.radians(angle)

    def project(x, y, z):
        u = x + y * scale * np.cos(theta)
        v = z + y * scale * np.sin(theta)
        return u, v

    # 1. 绘制底座侧面（先画侧面被遮挡的部分，再画顶面）
    # 右侧面
    p1 = project(width, 0, 0)
    p2 = project(width, height, 0)
    p3 = project(width, height, -depth)
    p4 = project(width, 0, -depth)
    ax.add_patch(Polygon([p1, p2, p3, p4], facecolor=side_color, edgecolor=edge_color))

    # 前侧面
    p1 = project(0, 0, 0)
    p2 = project(width, 0, 0)
    p3 = project(width, 0, -depth)
    p4 = project(0, 0, -depth)
    ax.add_patch(Polygon([p1, p2, p3, p4], facecolor=side_color, edgecolor=edge_color))

    # 2. 绘制顶面网格 (拼贴法)
    # 我们一个格子一个格子画，这样才有网格线
    for i in range(nx):
        for j in range(ny):
            x = i * step
            y = j * step

            # 每个格子的4个点 (z=0)
            v_local = [(x, y, 0), (x + step, y, 0), (x + step, y + step, 0), (x, y + step, 0)]
            v_proj = [project(*v) for v in v_local]

            ax.add_patch(Polygon(v_proj, facecolor=top_color, edgecolor=edge_color, linewidth=0.5))

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('grid_block.svg', format='svg', transparent=True, bbox_inches='tight')
    plt.show()


draw_grid_block_svg()