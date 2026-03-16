import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置学术字体和支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # 支持中文 (Windows用SimHei, Mac用Arial Unicode MS)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix' # 公式字体风格类似于 Times New Roman
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off') # 隐藏坐标轴

# 颜色定义
COLOR_GRID_EDGE = '#adb5bd'
COLOR_GRID_FACE = '#f8f9fa'
COLOR_ROW_HL = '#ffe3e3'
COLOR_ROW_EDGE = '#fa5252'
COLOR_COL_HL = '#d0ebff'
COLOR_COL_EDGE = '#339af0'
COLOR_CROSS_HL = '#eebefa'
COLOR_CROSS_EDGE = '#be4bdb'
COLOR_TEXT = '#212529'

def draw_grid(x_start, y_start, rows, cols, cell_size=0.6, facecolor=COLOR_GRID_FACE):
    """绘制特征网格"""
    for r in range(rows):
        for c in range(cols):
            rect = patches.Rectangle((x_start + c*cell_size, y_start + r*cell_size),
                                     cell_size, cell_size,
                                     linewidth=1.2, edgecolor=COLOR_GRID_EDGE, facecolor=facecolor)
            ax.add_patch(rect)

def draw_arrow(x_start, y_start, dx, dy, text="", color='#495057'):
    """绘制带文字的箭头"""
    ax.annotate("", xy=(x_start+dx, y_start+dy), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5, shrinkA=0, shrinkB=0))
    if text:
        # 文字居中在线的上方
        ax.text(x_start + dx/2, y_start + dy/2 + 0.2, text,
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)

# ==========================================
# (a) 左图：传统 2D GAP
# ==========================================
ax.text(4, 9, "(a) 传统 2D 全局平均池化 (2D GAP)", ha='center', fontsize=16, fontweight='bold', color=COLOR_TEXT)
ax.text(4, 8.4, "缺陷：二维坐标信息不可逆流失", ha='center', fontsize=12, color=COLOR_ROW_EDGE, fontweight='bold')

# 输入特征网格 (5x5)
draw_grid(1.5, 4, 5, 5)
ax.text(3, 3.5, "Input Feature", ha='center', fontsize=12, fontweight='bold')
ax.text(3, 3.0, r"$C \times H \times W$", ha='center', fontsize=12, color='gray')

# 2D GAP 箭头
draw_arrow(4.8, 5.5, 2.0, 0, text="2D GAP", color='#6c757d')

# 输出标量 (1x1)
draw_grid(7.2, 5.2, 1, 1, cell_size=0.6, facecolor='#e9ecef')
ax.text(7.5, 4.7, "Output", ha='center', fontsize=12, fontweight='bold')
ax.text(7.5, 4.2, r"$C \times 1 \times 1$", ha='center', fontsize=12, color='gray')
ax.text(7.5, 3.7, "(无序标量)", ha='center', fontsize=11, color='gray')

# 中间分割线
ax.plot([9.5, 9.5], [1, 9.5], color='#dee2e6', linestyle='--', linewidth=2)

# ==========================================
# (b) 右图：方向解耦的 1D 池化
# ==========================================
ax.text(14.5, 9, "(b) 方向解耦与正交交叉定位体系", ha='center', fontsize=16, fontweight='bold', color=COLOR_TEXT)
ax.text(14.5, 8.4, "优势：保留正交维度的精确位置坐标", ha='center', fontsize=12, color='#2b8a3e', fontweight='bold')

# 核心输入特征网格 (5x5)
draw_grid(11.5, 4, 5, 5)

# 高亮行 (Y轴坐标 y 处) - 索引为 2 (从下往上)
rect_row = patches.Rectangle((11.5, 4 + 2*0.6), 5*0.6, 0.6, linewidth=2, edgecolor=COLOR_ROW_EDGE, facecolor=COLOR_ROW_HL, alpha=0.9)
ax.add_patch(rect_row)
ax.text(11.2, 4 + 2.5*0.6, r"$y$", ha='right', va='center', fontsize=14, color=COLOR_ROW_EDGE, fontweight='bold')

# 高亮列 (X轴坐标 x 处) - 索引为 3 (从左往右)
rect_col = patches.Rectangle((11.5 + 3*0.6, 4), 0.6, 5*0.6, linewidth=2, edgecolor=COLOR_COL_EDGE, facecolor=COLOR_COL_HL, alpha=0.9)
ax.add_patch(rect_col)
ax.text(11.5 + 3.5*0.6, 7.2, r"$x$", ha='center', va='bottom', fontsize=14, color=COLOR_COL_EDGE, fontweight='bold')

# 交叉点
rect_cross = patches.Rectangle((11.5 + 3*0.6, 4 + 2*0.6), 0.6, 0.6, linewidth=2.5, edgecolor=COLOR_CROSS_EDGE, facecolor=COLOR_CROSS_HL)
ax.add_patch(rect_cross)

ax.text(13, 3.5, "Input Feature", ha='center', fontsize=12, fontweight='bold')
ax.text(13, 3.0, r"$C \times H \times W$", ha='center', fontsize=12, color='gray')

# 1. 提取 Height Vector (向右箭头)
draw_arrow(14.8, 5.5, 1.5, 0, text="Pool 1D", color=COLOR_ROW_EDGE)

# 垂直网格 (5x1)
draw_grid(16.8, 4, 5, 1)
# 高亮对应的行
rect_pool_y = patches.Rectangle((16.8, 4 + 2*0.6), 0.6, 0.6, linewidth=2, edgecolor=COLOR_ROW_EDGE, facecolor=COLOR_ROW_HL)
ax.add_patch(rect_pool_y)

ax.text(17.8, 5.5, r"$z_c^h(y)$", ha='left', va='center', fontsize=14, color=COLOR_TEXT)
ax.text(17.1, 3.5, "Height Vector", ha='center', fontsize=12, fontweight='bold')
ax.text(17.1, 3.0, r"$C \times H \times 1$", ha='center', fontsize=12, color='gray')

# 2. 提取 Width Vector (向下箭头)
draw_arrow(13.6, 3.7, 0, -1.2, text="Pool 1D", color=COLOR_COL_EDGE)

# 水平网格 (1x5)
draw_grid(11.5, 1.2, 1, 5)
# 高亮对应的列
rect_pool_x = patches.Rectangle((11.5 + 3*0.6, 1.2), 0.6, 0.6, linewidth=2, edgecolor=COLOR_COL_EDGE, facecolor=COLOR_COL_HL)
ax.add_patch(rect_pool_x)

ax.text(13.6, 0.7, r"$z_c^w(x)$", ha='center', va='top', fontsize=14, color=COLOR_TEXT)
ax.text(10.2, 1.5, "Width Vector", ha='right', va='center', fontsize=12, fontweight='bold')
ax.text(10.2, 1.1, r"$C \times 1 \times W$", ha='right', va='center', fontsize=12, color='gray')

# 自动调整并保存
plt.tight_layout()
plt.savefig("Fig2_4_Pooling_Mechanism.pdf", dpi=300, bbox_inches='tight')
plt.savefig("Fig2_4_Pooling_Mechanism.png", dpi=300, bbox_inches='tight')
print("图片已成功生成：Fig2_4_Pooling_Mechanism.pdf 和 .png")
# plt.show() # 取消注释可以在窗口预览