# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os  # 导入操作系统相关库，用于文件路径处理
import numpy as np  # 导入numpy库，用于数值计算和数组操作
from typing import Optional  # 类型提示，可选参数支持
from plyfile import PlyData, PlyElement  # ply文件读写库


def save_points_to_ply(filename: str, points: np.ndarray,
                       normals: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None,
                       labels: Optional[np.ndarray] = None,
                       labels_2: Optional[np.ndarray] = None,
                       text: bool = False):
    """
    保存点云数据到PLY文件，支持法线、颜色、标签等附加信息。
    参数：
        filename: 保存的文件路径
        points: 点云坐标，shape=(N,3)
        normals: 法线，shape=(N,3)，可选
        colors: 颜色，shape=(N,3)，可选，uint8类型
        labels: 标签，shape=(N,1)或(N,)，可选
        labels_2: 第二类标签，shape=(N,1)或(N,)，可选
        text: 是否以文本格式保存PLY（默认为False，二进制）
    """
    # 构建点云属性列表和类型
    point_cloud = [points]
    point_cloud_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if normals is not None:
        point_cloud.append(normals)
        point_cloud_types += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    if colors is not None:
        point_cloud.append(colors)
        point_cloud_types += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if labels is not None:
        point_cloud.append(labels)
        point_cloud_types += [('label', 'u1')]
    if labels_2 is not None:
        point_cloud.append(labels_2)
        point_cloud_types += [('label_2', 'u1')]
    # 拼接所有属性为一个二维数组
    point_cloud = np.concatenate(point_cloud, axis=1)

    # 构建结构化数组，每个点为一个元组
    vertices = [tuple(p) for p in point_cloud]
    structured_array = np.array(vertices, dtype=point_cloud_types)
    el = PlyElement.describe(structured_array, 'vertex')  # 构建PLY元素

    folder = os.path.dirname(filename)  # 获取文件夹路径
    if not os.path.exists(folder):  # 如果文件夹不存在，则创建
        os.makedirs(folder)
    PlyData([el], text).write(filename)  # 写入PLY文件
