# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch  # 导入PyTorch库，用于张量操作和深度学习
from thsolver import Dataset  # 导入项目自定义的数据集基类
from ocnn.octree import Points  # 导入八叉树点云结构，用于点云数据封装
from ocnn.dataset import CollateBatch  # 导入批量数据拼接工具

from .utils import ReadPly, Transform  # 导入点云文件读取和通用预处理工具
import os  # 导入操作系统相关库，用于文件路径处理

class ShapeNetTransform(Transform):
    """
    ShapeNet 数据集专用的预处理变换类，继承自通用 Transform。
    主要负责将原始点云样本转换为神经网络可用的格式。
    """
    def preprocess(self, sample: dict, idx: int):
        """
        对单个样本进行预处理，包括：
        - 点云坐标、法线、标签等转换为 float 类型的 torch 张量
        - 封装为 Points 对象，便于后续模型输入
        - 支持多标签（labels, labels_2）
        - 可选归一化（注释掉了，实际归一化可通过 bbox 和 scale 参数实现）
        """
        xyz = torch.from_numpy(sample['points']).float()  # 点云坐标
        normal = torch.from_numpy(sample['normals']).float()  # 法线




        labels = torch.from_numpy(sample['labels']).float()  # 主标签





        # labels_2 = torch.from_numpy(sample['labels_2']).float()  # 辅助标签
        # 封装为 Points 对象，labels/labels_2 需升维以适配 Points 接口
        points = Points(xyz, normal)

        # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
        # bbmin, bbmax = points.bbox()
        # points.normalize(bbmin, bbmax, scale=0.8)
        # 归一化功能已注释，如需归一化可取消注释
        return {'points': points}  # 返回封装后的点云对象字典





def load_tool_params(filename, tool_params_dir):
    tool_param_file = os.path.join(tool_params_dir, filename.replace('.ply', '.txt'))
    with open(tool_param_file, 'r') as f:
        # 读取第一行，跳过第一个字段（通常是文件名），取后面四个浮点数作为刀具参数
        tool_params = list(map(float, f.readline().split()[1:]))
    return torch.tensor(tool_params, dtype=torch.float32)  # 返回为float32张量



def get_seg_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  #TODO: 修改labels的读取方式
  read_ply = ReadPly(has_normal=True, has_label=True)
  collate_batch = CollateBatch(merge_points=True)


  # 创建数据集
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, take=flags.take )

  return dataset, collate_batch