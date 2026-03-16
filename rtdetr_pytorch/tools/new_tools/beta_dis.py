import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# 1. 强制设置无错字的中文字体 (Windows 推荐使用 'Microsoft YaHei' 或 'SimHei')
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 解决负号 '-' 显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

# 为了确保公式能正常显示
mpl.rcParams['mathtext.fontset'] = 'stix'
# 如果使用衬线字体会导致中文变方块，可以注释掉下面这行，或者保持默认
# mpl.rcParams['font.family'] = 'serif'

# 2. 生成高密度特征网格 (让曲面更平滑)
x = np.linspace(-0.99, 0.99, 150)
y = np.linspace(-0.99, 0.99, 150)
X, Y = np.meshgrid(x, y)

# 3. 严格计算基础中心度 (你代码里的逻辑)
min_lr = 1 - np.abs(X)
max_lr = 1 + np.abs(X)
min_tb = 1 - np.abs(Y)
max_tb = 1 + np.abs(Y)
C_raw = np.sqrt((min_lr / max_lr) * (min_tb / max_tb))

# 4. 动态 Beta 调制
C_normal = np.power(C_raw, 1.0)
C_tiny = np.power(C_raw, 0.2)

# 5. 开始绘制 3D 宽幅图表
fig = plt.figure(figsize=(14, 6))


# ======== 定义统一的 3D 坐标轴高级样式 ========
def style_3d_ax(ax):
    # 使底板和侧板完全透明，去除默认的灰色墙壁
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    # 将网格线调成低调的高级灰虚线
    ax.grid(color='grey', linestyle='-.', linewidth=0.3, alpha=0.4)

    # 设置坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.05)  # 顶部留一点呼吸空间

    # 设置完美观测视角 (仰角28度，方位角-45度)
    ax.view_init(elev=28, azim=-45)

    # 简化刻度，避免画面杂乱
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([0, 0.5, 1.0])


# ======== (a) 左图：较大目标 ========
ax1 = fig.add_subplot(121, projection='3d')
# 主曲面 (开启抗锯齿，使用柔和的蓝色渐变)
surf1 = ax1.plot_surface(X, Y, C_normal, cmap=cm.Blues, linewidth=0, antialiased=True, alpha=0.85)
# 底部投影 (学术制图的灵魂操作)
cset1 = ax1.contourf(X, Y, C_normal, zdir='z', offset=0, cmap=cm.Blues, alpha=0.6)

ax1.set_title(r'(a) 较大目标 ($\beta \approx 1.0$): 尖峰金字塔分布', fontsize=17, pad=20)
# ax1.set_xlabel('相对 X 坐标 (X / w)', fontsize=12, labelpad=5)
# ax1.set_ylabel('相对 Y 坐标 (Y / h)', fontsize=12, labelpad=5)
# ax1.set_zlabel('中心度得分 (Centerness Score)', fontsize=12, labelpad=5)
style_3d_ax(ax1)

# ======== (b) 右图：极小目标 ========
ax2 = fig.add_subplot(122, projection='3d')
# 主曲面 (使用极具视觉冲击力的红色渐变)
surf2 = ax2.plot_surface(X, Y, C_tiny, cmap=cm.Reds, linewidth=0, antialiased=True, alpha=0.85)
# 底部投影
cset2 = ax2.contourf(X, Y, C_tiny, zdir='z', offset=0, cmap=cm.Reds, alpha=0.6)

ax2.set_title(r'(b) 微小目标 ($\beta \rightarrow 0.2$): 扩展平顶分布', fontsize=17,  pad=20)
# ax2.set_xlabel('相对 X 坐标 (X / w)', fontsize=12, labelpad=5)
# ax2.set_ylabel('相对 Y 坐标 (Y / h)', fontsize=12, labelpad=5)
# ax2.set_zlabel('中心度得分 (Centerness Score)', fontsize=12, labelpad=5)
style_3d_ax(ax2)

# 6. 保存输出
plt.tight_layout()
plt.savefig('Fig_3D_Adaptive_Centerness_Pro.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Fig_3D_Adaptive_Centerness_Pro.png', dpi=300, bbox_inches='tight')
print("高级版 3D 分布图生成成功！快去看看 PDF 吧！")