# --------------------------------------------------------  # 文件头，版权声明和作者信息
# Octree-based Sparse Convolutional Neural Networks         # 项目名称
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>   # 版权信息
# Licensed under The MIT License [see LICENSE for details]   # 许可证信息
# Written by Peng-Shuai Wang                                # 作者信息
# --------------------------------------------------------

import os  # 导入os模块，用于文件路径和文件操作
import torch  # 导入torch库，用于张量和深度学习
import numpy as np  # 导入numpy库，用于数值计算
from thsolver import Dataset  # 从thsolver包导入Dataset类
from ocnn.octree import Points  # 从ocnn.octree模块导入Points类
from ocnn.dataset import CollateBatch  # 从ocnn.dataset模块导入CollateBatch类

from .utils import Transform  # 从当前包导入Transform类

# label_name_mapping 用于将语义KITTI数据集中的标签ID映射为对应的类别名称
label_name_mapping = {
    0: 'unlabeled',  # 未标记类别
    1: 'outlier',  # 异常点
    10: 'car',  # 汽车
    11: 'bicycle',  # 自行车
    13: 'bus',  # 公交车
    15: 'motorcycle',  # 摩托车
    16: 'on-rails',  # 轨道上的物体
    18: 'truck',  # 卡车
    20: 'other-vehicle',  # 其他车辆
    30: 'person',  # 行人
    31: 'bicyclist',  # 骑自行车的人
    32: 'motorcyclist',  # 骑摩托车的人
    40: 'road',  # 道路
    44: 'parking',  # 停车场
    48: 'sidewalk',  # 人行道
    49: 'other-ground',  # 其他地面
    50: 'building',  # 建筑物
    51: 'fence',  # 围栏
    52: 'other-structure',  # 其他结构
    60: 'lane-marking',  # 车道标记
    70: 'vegetation',  # 植被
    71: 'trunk',  # 树干
    72: 'terrain',  # 地形
    80: 'pole',  # 杆
    81: 'traffic-sign',  # 交通标志
    99: 'other-object',  # 其他物体
    252: 'moving-car',  # 移动车辆
    253: 'moving-bicyclist',  # 移动骑自行车的人
    254: 'moving-person',  # 移动行人
    255: 'moving-motorcyclist',  # 移动骑摩托车的人
    256: 'moving-on-rails',  # 移动轨道物体
    257: 'moving-bus',  # 移动公交车
    258: 'moving-truck',  # 移动卡车
    259: 'moving-other-vehicle'  # 移动其他车辆
}

# kept_labels 定义了在训练和评估中保留的类别名称（共19类）
kept_labels = [
    'road',  # 道路
    'sidewalk',  # 人行道
    'parking',  # 停车场
    'other-ground',  # 其他地面
    'building',  # 建筑物
    'car',  # 汽车
    'truck',  # 卡车
    'bicycle',  # 自行车
    'motorcycle',  # 摩托车
    'other-vehicle',  # 其他车辆
    'vegetation',  # 植被
    'trunk',  # 树干
    'terrain',  # 地形
    'person',  # 行人
    'bicyclist',  # 骑自行车的人
    'motorcyclist',  # 骑摩托车的人
    'fence',  # 围栏
    'pole',  # 杆
    'traffic-sign'  # 交通标志
]


def get_label_map():
  """
  构建标签映射表，将原始标签ID映射为训练用的连续类别ID（1~19），
  其余标签映射为0（忽略类别）。
  返回: 长度为260的numpy数组，索引为原始标签ID，值为新类别ID。
  """
  num_classes = len(kept_labels)  # 统计保留类别数量（19类）
  label_ids = list(range(1, num_classes + 1))  # 生成类别ID列表[1,2,...,19]
  label_dict = dict(zip(kept_labels, label_ids))  # 构建类别名称到ID的映射字典

  label_map = np.zeros(260)  # 初始化标签映射表，长度260，全部为0
  for idx, name in label_name_mapping.items():  # 遍历所有原始标签ID和名称
    name = name.replace('moving-', '')  # 将移动类别名称映射为静态类别
    label_map[idx] = label_dict.get(name, 0)  # 若在保留类别中则赋值新ID，否则为0
  return label_map  # 返回标签映射表


class KittiTransform(Transform):  # 继承自Transform类，专用于KITTI数据预处理

  def __init__(self, flags):
    super().__init__(flags)  # 调用父类初始化

    self.scale_factor = 100  # 设置点云归一化的缩放因子
    self.label_map = get_label_map()  # 获取标签映射表

  def preprocess(self, sample, idx=None):
    # 获取输入点云和密度
    xyz = sample['points'][:, :3]  # 提取点的xyz坐标
    density = sample['points'][:, 3:]  # 提取点的密度信息

    # 对xyz坐标进行归一化，使其分布在[-1, 1]区间
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0  # 计算点云中心
    xyz = (xyz - center) / self.scale_factor  # 坐标减中心后缩放

    # 标签重映射，将原始标签转换为训练类别ID
    labels = sample['labels']  # 获取原始标签
    labels = self.label_map[labels & 0xFFFF].astype(np.float32)  # 映射并转为float32

    points = Points(
        torch.from_numpy(xyz),  # 转为torch张量
        None,  # 占位（未使用的特征）
        torch.from_numpy(density),  # 密度转为torch张量
        torch.from_numpy(labels).unsqueeze(1)  # 标签转为torch张量并升维
    )
    return {'points': points}  # 返回处理后的点云对象


def read_file(filename):
  points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)  # 读取点云数据，按4维重塑

  label_name = filename.replace('velodyne', 'labels').replace('.bin', '.label')  # 构造标签文件名
  if os.path.exists(label_name):  # 判断标签文件是否存在
    labels = np.fromfile(label_name, dtype=np.int32).reshape(-1)  # 读取标签数据
  else:
    labels = np.zeros((points.shape[0],), dtype=np.int32)  # 若无标签则全为0

  return {'points': points, 'labels': labels}  # 返回点云和标签字典


def get_kitti_dataset(flags):
  transform = KittiTransform(flags)  # 实例化KittiTransform类
  collate_batch = CollateBatch(merge_points=True)  # 实例化CollateBatch类
  dataset = Dataset(flags.location, flags.filelist, transform,  # 实例化Dataset类
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_batch  # 返回数据集和批处理器
