import os
import numpy as np
import pandas as pd
import argparse
import concurrent.futures
import time

# 旋转矩阵转6D表示
def matrix_to_6d(matrix):
    """
    将3x3旋转矩阵转换为6D表示（提取前两列并展平）
    参数：matrix: shape=(3,3)
    返回：6D向量，shape=(6,)
    """
    a1 = matrix[:, 0]
    a2 = matrix[:, 1]
    six_d = np.concatenate([a1, a2])
    return six_d

# 批量处理主函数
def batch_rotmat_to_6d_txt(root_folder):
    """
    批量遍历root_folder下所有B文件夹，处理target文件夹下的csv文件，
    转换为6D表示并保存为txt到B文件夹下，命名为Bxx_label.txt
    """
    # 遍历所有B文件夹
    for b_name in os.listdir(root_folder):
        b_path = os.path.join(root_folder, b_name)
        if not os.path.isdir(b_path):
            continue  # 跳过非文件夹
        target_path = os.path.join(b_path, 'target')
        if not os.path.isdir(target_path):
            print(f"{b_name} 下无 target 文件夹，跳过")
            continue
        # 查找csv文件
        csv_files = [f for f in os.listdir(target_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"{b_name}/target 下无csv文件，跳过")
            continue
        csv_file = os.path.join(target_path, csv_files[0])  # 默认取第一个csv
        # 读取csv
        df = pd.read_csv(csv_file)
        mats = []
        for i in range(len(df)):
            row = df.iloc[i].values.astype(float)
            if row.shape[0] != 9:
                print(f"{b_name} 第{i}行不是9个元素，跳过")
                continue
            mat = row.reshape(3, 3)
            mats.append(mat)
        # 转换为6D表示
        sixd_list = [matrix_to_6d(mat) for mat in mats]
        # 保存为txt，使用np.savetxt自动处理符号和精度，避免丢失符号
        txt_path = os.path.join(b_path, f"{b_name}_label.txt")
        np.savetxt(txt_path, np.array(sixd_list), fmt="%.8f")
        print(f"已保存 {txt_path}，共 {len(sixd_list)} 行")

def process_b_folder(b_name, root_folder):
    """
    多线程处理单个B文件夹，读取target/csv，转换为6D表示并保存为Bxx_label.txt
    """
    b_path = os.path.join(root_folder, b_name)
    if not os.path.isdir(b_path):
        return f"{b_name} 不是文件夹，跳过"
    target_path = os.path.join(b_path, 'target')
    if not os.path.isdir(target_path):
        return f"{b_name} 下无 target 文件夹，跳过"
    csv_files = [f for f in os.listdir(target_path) if f.endswith('.csv')]
    if not csv_files:
        return f"{b_name}/target 下无csv文件，跳过"
    csv_file = os.path.join(target_path, csv_files[0])
    try:
        df = pd.read_csv(csv_file)
        mats = []
        for i in range(len(df)):
            row = df.iloc[i].values.astype(float)
            if row.shape[0] != 9:
                continue
            mat = row.reshape(3, 3)
            mats.append(mat)
        sixd_list = [matrix_to_6d(mat) for mat in mats]
        txt_path = os.path.join(b_path, f"{b_name}_label.txt")
        with open(txt_path, 'w') as f:
            for sixd in sixd_list:
                f.write(' '.join(f"{x:.8f}" for x in sixd) + '\n')
        return f"{b_name} 已保存 {txt_path}，共 {len(sixd_list)} 行"
    except Exception as e:
        return f"{b_name} 处理异常: {e}"

# 多线程主函数
def batch_rotmat_to_6d_txt_multithread(root_folder, max_workers=8):
    """
    多线程批量处理所有B文件夹，加速6D表示生成
    """
    b_names = [b for b in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, b))]
    print(f"共发现 {len(b_names)} 个B文件夹，开始多线程处理...")
    start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_b = {executor.submit(process_b_folder, b, root_folder): b for b in b_names}
        for future in concurrent.futures.as_completed(future_to_b):
            res = future.result()
            results.append(res)
            print(res)
    end_time = time.time()
    print(f"多线程处理完成，总耗时 {end_time - start_time:.2f} 秒")
    return results

if __name__ == "__main__":
    # 命令行参数解析，支持自定义root_folder
    parser = argparse.ArgumentParser(description="批量旋转矩阵转6D表示（多线程加速版）")
    parser.add_argument('--root_folder', type=str, default="F:\\keep",
                        help='顶层数据文件夹路径，默认为当前脚本所在目录')
    parser.add_argument('--max_workers', type=int, default=8, help='最大线程数，默认8')
    args = parser.parse_args()
    batch_rotmat_to_6d_txt_multithread(args.root_folder, args.max_workers)
