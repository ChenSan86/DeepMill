# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import numpy as np
from plyfile import PlyData
import os


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
    #TODO change plyData.read
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
      #TODO这里不读点云的label
    if self.has_label:

        # print("\n"*10)
        # print(filename)
        basename = os.path.basename(filename)
        # 去掉后缀和_collision_detection
        model_name = basename.replace('_collision_detection.ply', '')
        # print(basename)
        # print(model_name)
        label = read_six_dim_vector(model_name)
        # print(label)

        output['labels'] = np.array(label).astype(np.float32)


    #   # 如果需要标签，则读取标签信息
    #   label = vtx['label']
    #   output['labels'] = label.astype(np.int32)
    # # 额外读取label_2标签
    #   label_2 = vtx['label_2']
    #   output['labels_2'] = label_2.astype(np.int32)

    #TODO 读取一个六维向量，读取方式为，去data/filelist/models_test.txt;models_train_val.txt找到对应的模型名称的行，然后每一行后六个浮点数即为要读取的六维向量
    # print("+"*10)
    # print(output)
    # print("+" * 10)
    return output
'''++++++++++
{'points': array([[-0.12943 ,  0.427911,  0.377951],
       [-0.08304 ,  0.43407 ,  0.391642],
       [-0.12943 ,  0.434349,  0.416061],
       ...,
       [ 0.446988,  0.338269,  0.002548],
       [-0.14883 , -0.741827,  0.120998],
       [ 0.049073, -0.121524, -0.35637 ]], dtype=float32), 'normals': array([[ 6.69552e-01, -6.85796e-01,  2.85279e-01],
       [ 3.54418e-01, -3.79133e-01,  8.54778e-01],
       [ 6.87689e-01, -2.30610e-02,  7.25639e-01],
       ...,
       [ 5.66820e-01,  4.00000e-06, -8.23842e-01],
       [-2.77699e-01, -8.57687e-01, -4.32731e-01],
       [-4.99923e-01,  0.00000e+00,  8.66070e-01]], dtype=float32), 'labels': array([ 0.92666 , -0.017757, -0.375481, -0.017757,  0.995701, -0.090909],
      dtype=float32)}
++++++++++
       '''


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
  r''' Wraps :class:`ocnn.data_2.0.Transform` for convenience.
  '''

  def __init__(self, flags):
    super().__init__(**flags)
    self.flags = flags


import os

def read_six_dim_vector(model_name, filelist_dir='data_2.0/filelist'):
    """
    根据模型名称，查找文件并读取六维向量
    :param model_name: 模型文件名（如 models/xxx.ply）
    :param filelist_dir: 文件列表目录
    :return: 六维向量（list[float]），未找到则返回None
    """
    file_paths = [
        os.path.join(filelist_dir, 'models_test.txt'),
        os.path.join(filelist_dir, 'models_train_val.txt')
    ]
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if model_name in line:
                    parts = line.strip().split()
                    # 最后六个浮点数
                    try:
                        vector = [float(x) for x in parts[-6:]]
                        return vector
                    except Exception:
                        pass
    return None