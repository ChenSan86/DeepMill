# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm


def read_file(filename):
  points = np.fromfile(filename, dtype=np.uint8)
  return torch.from_numpy(points)   # convert it to torch.tensor


class Dataset(torch.utils.data.Dataset):

  def __init__(self, root, filelist, transform, read_file=read_file,
               in_memory=False, take: int = -1):
    super(Dataset, self).__init__()
    self.root = root
    self.filelist = filelist
    self.transform = transform
    self.in_memory = in_memory
    self.read_file = read_file
    self.take = take  # 训练/测试时选取的样本数量

    self.filenames, self.labels, self.tool_params= self.load_filenames()  # 加载文件名、标签和刀具参数
    if self.in_memory:
      print('Load files into memory from ' + self.filelist)  # 打印加载信息
      self.samples = [self.read_file(os.path.join(self.root, f))
                      for f in tqdm(self.filenames, ncols=80, leave=False)]  # 预加载所有样本到内存

  def __len__(self):
    return len(self.filenames)  # 返回样本数量

  def __getitem__(self, idx):
    sample = (self.samples[idx] if self.in_memory else
              self.read_file(os.path.join(self.root, self.filenames[idx])))  # 获取单个样本
    output = self.transform(sample, idx)  # 数据增强和octree构建
    output['label'] = self.labels[idx]  # 添加标签
    output['filename'] = self.filenames[idx]  # 添加文件名
    # 这里确保刀具参数添加到输出中
    output['tool_params'] = self.tool_params[idx]  # 假设在加载数据时已经填充
    return output  # 返回样本字典

  def load_filenames(self):
    filenames, labels, tool_params = [], [], []  # 初始化列表
    with open(self.filelist) as fid:
      lines = fid.readlines()  # 读取所有行
    for line in lines:
      tokens = line.split()  # 按空格分割
      filename = tokens[0].replace('\\', '/')  # 获取文件名并规范路径分隔符

      # 获取tool_params中的后4位
      if len(tokens) >= 2:
        label = tokens[1]  # 获取标签
        # 读取tool_params中的后4位并进行处理，假设tool_params是4维向量
        tool_param = tokens[-4:]  # 获取最后4位作为刀具参数
      else:
        label = 0  # 默认���签为0

      filenames.append(filename)  # 添加文件名
      labels.append(int(label))  # 添加标签
      tool_params.append(tool_param)  # 添加刀具参数

    num = len(filenames)  # 样本总数
    if self.take > num or self.take < 1:
      self.take = num  # 修正take参数
    result = (filenames[:self.take], labels[:self.take], tool_params[:self.take])

    return result  # 返回指定数量的文件名、标签和刀具参数
