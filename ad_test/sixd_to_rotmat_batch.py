import numpy as np
import pandas as pd

# 工具函数1：接收一个csv文件，返回一个3x3矩阵（支持负号）
def csv_to_matrix(csv_path):
    """
    读取csv文件，返回第一个3x3旋转矩阵（支持负号）
    参数：csv_path: csv文件路径
    返回：3x3 numpy数组
    """
    df = pd.read_csv(csv_path)
    # 取第一行，转换为float，reshape为3x3
    row = df.iloc[0].values.astype(float)
    if row.shape[0] != 9:
        raise ValueError("CSV文件首行不是9个元素，无法转换为3x3矩阵")
    mat = row.reshape(3, 3)
    return mat

# 工具函数2：接收一个3x3矩阵，转为6维向量表示
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

# 工具函数3：接收一个6维向量，还原为3x3矩阵
def sixd_to_matrix(six_d):
    """
    将6D向量还原为3x3旋转矩阵
    参数：six_d: shape=(6,)
    返回：旋转矩阵，shape=(3,3)
    """
    a1 = six_d[:3]
    a2 = six_d[3:]
    b1 = a1 / np.linalg.norm(a1)
    proj = np.dot(b1, a2) * b1
    a2_ortho = a2 - proj
    b2 = a2_ortho / np.linalg.norm(a2_ortho)
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=1)
    return matrix

def geodesic_loss(R1, R2):
    """
    计算两个旋转矩阵之间的Geodesic Loss（弧度制）
    参数：
        R1: shape=(3,3) 或 (N,3,3)，第一个旋转矩阵或批量
        R2: shape=(3,3) 或 (N,3,3)，第二个旋转矩阵或批量
    返回：
        loss: 标量或shape=(N,)的numpy数组，表示每对旋转的弧度距离
    """
    # 支持批量输入
    R1 = np.array(R1)
    R2 = np.array(R2)
    if R1.ndim == 2:
        R1 = R1[None, ...]
    if R2.ndim == 2:
        R2 = R2[None, ...]
    # 计算相对旋转矩阵
    R_rel = np.matmul(R1, np.transpose(R2, (0,2,1)))
    # 计算迹
    trace = np.trace(R_rel, axis1=1, axis2=2)
    # 计算夹角（弧度），防止数值溢出
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta.shape[0] == 1:
        return theta[0]
    return theta
