# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import numpy as np
from plyfile import PlyData


# ReadPly类用于读取PLY格式的点云文件，并根据参数提取法线、颜色、标签等信息
class ReadPly:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    # 初始化，指定是否读取法线、颜色、标签
    self.has_normal = has_normal  # 是否读取法线信息
    self.has_color = has_color    # 是否读取颜色信息
    self.has_label = has_label    # 是否读取标签信息

  def __call__(self, filename: str):
    # 读取PLY文件，返回包含点云、法线、颜色、标签等信息的字典
    plydata = PlyData.read(filename)  # 读取PLY文件
    vtx = plydata['vertex']           # 获取顶点数据

    output = dict()                   # 用于存储输出结果
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)  # 组装点坐标
    output['points'] = points.astype(np.float32)               # 转为float32类型
    if self.has_normal:
      # 如果需要法线，则读取法线信息
      normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
      output['normals'] = normal.astype(np.float32)
    if self.has_color:
      # 如果需要颜色，则读取颜色信息
      color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
      output['colors'] = color.astype(np.float32)
    if self.has_label:
      # 如果需要标签，则读取标签信息
      label = vtx['label']
      output['labels'] = label.astype(np.int32)
    # 额外读取label_2标签
    label_2 = vtx['label_2']
    output['labels_2'] = label_2.astype(np.int32)
    return output


# ReadNpz类用于读取npz格式的点云数据文件，参数同ReadPly
class ReadNpz:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    # 初始化，指定是否读取法线、颜色、标签
    self.has_normal = has_normal  # 是否读取法线信息
    self.has_color = has_color    # 是否读取颜色信息
    self.has_label = has_label    # 是否读取标签信息

  def __call__(self, filename: str):
    # 读取npz文件，返回包含点云、法线、颜色、标签等信息的字典
    raw = np.load(filename)

    output = dict()
    output['points'] = raw['points'].astype(np.float32)  # 读取点云数据
    if self.has_normal:
      output['normals'] = raw['normals'].astype(np.float32)  # 读取法线数据
    if self.has_color:
      output['colors'] = raw['colors'].astype(np.float32)    # 读取颜色数据
    if self.has_label:
      output['labels'] = raw['labels'].astype(np.int32)      # 读取标签数���
    return output


class ReadFile:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.read_npz = ReadNpz(has_normal, has_color, has_label)
    self.read_ply = ReadPly(has_normal, has_color, has_label)

  def __call__(self, filename: str):
    func = {'npz': self.read_npz, 'ply': self.read_ply}
    suffix = filename.split('.')[-1]
    return func[suffix](filename)


class Transform(ocnn.dataset.Transform):
  r''' Wraps :class:`ocnn.data.Transform` for convenience.
  '''

  def __init__(self, flags):
    super().__init__(**flags)
    self.flags = flags
