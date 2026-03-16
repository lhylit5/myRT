import numpy as np
import matplotlib.pyplot as plt

# 全局学术字体设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def plot_cost_competition():
    # 生成网格数据 (X: Cls, Y: IoU)
    cls_scores = np.linspace(0.01, 1.0, 200)
    ious = np.linspace(0.01, 1.0, 200)
    X, Y = np.meshgrid(cls_scores, ious)

    # 1. 传统线性 Cost (越低越好，模拟原版加权求和)
    linear_cost = 0.5 * (1 - X) + 0.5 * (1 - Y)

    # 2. 本文的高阶对齐 Cost (越低越好，模拟高阶乘积)
    nonlinear_cost = 1 - (X * Y)

    # 定义两个竞争点
    pt_A = {'cls': 0.95, 'iou': 0.30, 'color': 'red'}
    pt_B = {'cls': 0.60, 'iou': 0.60, 'color': 'green'}

    # 计算两个点在两种机制下的实际 Cost
    cost_lin_A = 0.5 * (1 - pt_A['cls']) + 0.5 * (1 - pt_A['iou'])  # 0.375
    cost_lin_B = 0.5 * (1 - pt_B['cls']) + 0.5 * (1 - pt_B['iou'])  # 0.400

    cost_non_A = 1 - (pt_A['cls'] * pt_A['iou'])  # 0.715
    cost_non_B = 1 - (pt_B['cls'] * pt_B['iou'])  # 0.640

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ======= 左图：线性匹配代价 =======
    ax = axes[0]
    # 使用RdYlBu颜色映射，深蓝色代表低Cost(好)，深红色代表高Cost(差)
    CS1 = ax.contourf(X, Y, linear_cost, levels=20, cmap='RdYlBu', alpha=0.8)
    ax.contour(X, Y, linear_cost, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    # 标出 A 和 B (把字母标在点的前面)
    ax.scatter([pt_A['cls']], [pt_A['iou']], color=pt_A['color'], s=250, marker='o', edgecolors='black', zorder=5)
    ax.text(pt_A['cls'] - 0.08, pt_A['iou'] - 0.02, 'A', color='red', fontsize=20, fontweight='bold', zorder=6)

    ax.scatter([pt_B['cls']], [pt_B['iou']], color=pt_B['color'], s=250, marker='o', edgecolors='black', zorder=5)
    ax.text(pt_B['cls'] - 0.08, pt_B['iou'] - 0.02, 'B', color='green', fontsize=20, fontweight='bold', zorder=6)

    # 左上角文本框
    text_lin = (f"Cost(A) = {cost_lin_A:.3f}\n"
                f"Cost(B) = {cost_lin_B:.3f}\n"
                f"Result: A is Selected ")
    ax.text(0.04, 0.82, text_lin, fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

    ax.set_title("(a) Linear Cost", fontweight='bold')
    ax.set_xlabel("Classification Confidence")
    ax.set_ylabel("Bounding Box IoU")

    # ======= 右图：高阶任务对齐度量 =======
    ax = axes[1]
    CS2 = ax.contourf(X, Y, nonlinear_cost, levels=20, cmap='RdYlBu', alpha=0.8)
    ax.contour(X, Y, nonlinear_cost, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    # 标出 A 和 B (把字母标在点的前面)
    ax.scatter([pt_A['cls']], [pt_A['iou']], color=pt_A['color'], s=250, marker='o', edgecolors='black', zorder=5)
    ax.text(pt_A['cls'] - 0.08, pt_A['iou'] - 0.02, 'A', color='red', fontsize=20, fontweight='bold', zorder=6)

    ax.scatter([pt_B['cls']], [pt_B['iou']], color=pt_B['color'], s=250, marker='o', edgecolors='black', zorder=5)
    ax.text(pt_B['cls'] - 0.08, pt_B['iou'] - 0.02, 'B', color='green', fontsize=20, fontweight='bold', zorder=6)

    # 左上角文本框
    text_non = (f"Cost(A) = {cost_non_A:.3f}\n"
                f"Cost(B) = {cost_non_B:.3f}\n"
                f"Result: B is Selected")
    ax.text(0.04, 0.82, text_non, fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))

    ax.set_title("(b) Task-Aligned Cost (Ours)", fontweight='bold')
    ax.set_xlabel("Classification Confidence")
    ax.set_ylabel("Bounding Box IoU")

    plt.tight_layout()
    plt.savefig("Figure3_2_Cost_Competition.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_cost_competition()