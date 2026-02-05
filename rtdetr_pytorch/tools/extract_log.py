import os


def extract_test_results(input_file, output_file):
    """
    从训练日志中提取测试结果(AP, AR, best_stat)并保存到新文件
    """
    # 定义我们要提取的关键特征词
    keywords = [
        "IoU metric:",
        "Average Precision",
        "Average Recall",
        "best_stat"
    ]

    print(f"正在读取: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            # 写入一个文件头（可选）
            f_out.write(f"Extraction result from {input_file}:\n")
            f_out.write("=" * 50 + "\n")

            for line in f_in:
                line = line.strip()

                # 1. 筛选包含关键词的行
                if any(k in line for k in keywords):

                    # 2. 二次过滤：去掉那些带有进度条的行
                    # 特征：包含 "[" 和 "/" 和 "eta"，通常是 Process bar
                    if "[" in line and "/" in line and "eta:" in line:
                        continue

                    # 3. 既打印到屏幕，又写入文件
                    print(line)  # 打印
                    f_out.write(line + '\n')  # 写入

        print(f"\n✅ 成功！结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_file}，请检查路径是否正确。")


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 输入的日志文件名 (你之前的训练日志)
    INPUT_LOG = "E:\experiment\smallenhance_72epo\\train_areaweight.log"

    # 输出的日志文件名 (提取后的结果)
    OUTPUT_LOG = "test_results.log"
    # ===========================================

    extract_test_results(INPUT_LOG, OUTPUT_LOG)