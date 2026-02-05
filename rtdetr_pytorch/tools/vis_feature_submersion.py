import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 1. 输入图片路径
INPUT_IMG = '../configs/dataset/coco/val2017/000000025424.jpg'

# 2. 小目标中心点坐标 (x, y)
TARGET_CENTER = (397, 44)
TARGET_SIZE = 40  # 框的大小

# 3. 输出文件名
OUTPUT_FILE = 'Fig2_1_Feature_Submersion.png'


# ===========================================

def create_pixelated_view(img, stride, label):
    h, w = img.shape[:2]

    # 1. 模拟下采样
    h_feat, w_feat = h // stride, w // stride
    small = cv2.resize(img, (w_feat, h_feat), interpolation=cv2.INTER_LINEAR)

    # 2. 模拟“特征可视化” (Upsampling)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. 叠加网格
    overlay = pixelated.copy()
    for x in range(0, w, stride):
        cv2.line(overlay, (x, 0), (x, h), (200, 200, 200), 1)
    for y in range(0, h, stride):
        cv2.line(overlay, (0, y), (w, y), (200, 200, 200), 1)
    cv2.addWeighted(overlay, 0.3, pixelated, 0.7, 0, pixelated)

    # 4. 在目标位置画高亮框 (如果设置了坐标)
    if TARGET_CENTER is not None:
        cx, cy = TARGET_CENTER
        tl = (cx - TARGET_SIZE // 2, cy - TARGET_SIZE // 2)
        br = (cx + TARGET_SIZE // 2, cy + TARGET_SIZE // 2)

        if stride == 32:  # S5 层
            color = (0, 0, 255)  # Red
            thickness = 2
            cv2.rectangle(pixelated, tl, br, color, thickness)
            cv2.line(pixelated, tl, br, color, thickness)
            cv2.line(pixelated, (tl[0], br[1]), (br[0], tl[1]), color, thickness)
            cv2.putText(pixelated, "", (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            color = (0, 255, 0)  # Green
            cv2.rectangle(pixelated, tl, br, color, 2)

    # ================= 修改点 1：移出 if 块，保证无论是否有框都加底标 =================
    # 5. 添加文字标签 (改为图片外部下方)
    # 获取当前图像尺寸 (此时还是原图大小)
    curr_h, curr_w = pixelated.shape[:2]

    # 定义底部扩展的高度
    margin_height = 40

    # 给图片底部添加白色边框
    pixelated = cv2.copyMakeBorder(
        pixelated,
        0, margin_height, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)  # 白色背景
    )

    # 在新增加的区域写字
    # y 坐标 = 原高度 + 偏移量
    cv2.putText(pixelated, label, (20, curr_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    return pixelated


def main():
    if not os.path.exists(INPUT_IMG):
        print(f"错误：找不到图片 {INPUT_IMG}")
        return

    img = cv2.imread(INPUT_IMG)
    # img = img[100:600, 100:600] # 如果需要裁剪可取消注释

    # 生成三个阶段的图
    view_s3 = create_pixelated_view(img, 8, "Stage 3 (Stride=8): Clear")
    view_s4 = create_pixelated_view(img, 16, "Stage 4 (Stride=16): Blurry")
    view_s5 = create_pixelated_view(img, 32, "Stage 5 (Stride=32): Submerged")

    # ================= 修改点 2：根据生成图的高度创建分隔条 =================
    # 因为加上了底边框，现在图片高度变了，必须获取新的高度
    new_h = view_s3.shape[0]

    # 创建分隔条 (高度必须与 new_h 一致)
    separator = np.ones((new_h, 10, 3), dtype=np.uint8) * 255

    # 拼接
    final_img = np.hstack([view_s3, separator, view_s4, separator, view_s5])

    cv2.imwrite(OUTPUT_FILE, final_img)
    print(f"生成成功！请查看: {OUTPUT_FILE}")

    # 预览 (转换颜色空间用于 matplotlib 显示)
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()