# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import time
import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional


class AverageTracker:

  def __init__(self):
        # 初始化追踪器，分别用于存储各项指标的累计值和计数
        self.value = dict()  # 存储每个指标的累计值
        self.num = dict()    # 存储每个指标的累计次数
        self.max_len = 76    # 最大长度（可用于显示等）
        self.tick = time.time()      # 记录上一次更新时间
        self.start_time = time.time()# 记录追踪器启动时间

  def update(self, value: Dict[str, torch.Tensor], record_time: bool = True):
        r'''Update the tracker with the given value, which is called at the end of
        each iteration.
        '''
        # 更新追踪器，通常在每次迭代结束时调用
        # value: 包含各项指标的字典，key为指标名，value为Tensor
        # record_time: 是否记录本次迭代耗时

        if not value:
            return    # 输入为空，直接返回

        # 粗略记录本次迭代耗时
        if record_time:
            curr_time = time.time()
            value['time/iter'] = torch.Tensor([curr_time - self.tick]) # 计算本次迭代耗时
            self.tick = curr_time  # 更新时间戳

        # 更新累计值和计数
        for key, val in value.items():
            self.value[key] = self.value.get(key, 0) + val.detach()  # 累加指标值
            self.num[key] = self.num.get(key, 0) + 1                # 累加计数

  def average(self):
        # 计算各项指标的平均值
        return {key: val.item() / self.num[key] for key, val in self.value.items()}

  @torch.no_grad()
  def average_all_gather(self):
        r'''Average the tensors on all GPUs using all_gather, which is called at the
        end of each epoch.
        '''
        # 在多GPU环境下，使用all_gather收集所有GPU上的指标，并计算平均值
        for key, tensor in self.value.items():
            if not (isinstance(tensor, torch.Tensor) and tensor.is_cuda):
                continue  # 只收集GPU上的Tensor
            tensors_gather = [torch.ones_like(tensor)
                        for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
            tensors = torch.stack(tensors_gather, dim=0)
            self.value[key] = torch.mean(tensors)

  def log(self, epoch: int, summary_writer: Optional[SummaryWriter] = None,
          log_file: Optional[str] = None, msg_tag: str = '->', notes: str = '',
          print_time: bool = True, print_memory: bool = False):
    r'''Log the average value to the console, tensorboard and log file.
    '''
    if not self.value:
      return  # empty, return

    avg = self.average()
    msg = 'Epoch: %d' % epoch
    for key, val in avg.items():
      msg += ', %s: %.3f' % (key, val)
      if summary_writer is not None:
        summary_writer.add_scalar(key, val, epoch)

    # if the log_file is provided, save the log
    if log_file is not None:
      with open(log_file, 'a') as fid:
        fid.write(msg + '\n')

    # memory
    memory = ''
    if print_memory and torch.cuda.is_available():
      size = torch.cuda.memory_reserved()
      # size = torch.cuda.memory_allocated()
      memory = ', memory: {:.3f}GB'.format(size / 2**30)

    # time
    time_str = ''
    if print_time:
      curr_time = ', time: ' + datetime.now().strftime("%Y/%m/%d %H:%M:%S")
      duration = ', duration: {:.2f}s'.format(time.time() - self.start_time)
      time_str = curr_time + duration

    # other notes
    if notes:
      notes = ', ' + notes

    # concatenate all messages
    msg += memory + time_str + notes

    # split the msg for better display
    chunks = [msg[i:i+self.max_len] for i in range(0, len(msg), self.max_len)]
    msg = (msg_tag + ' ') + ('\n' + len(msg_tag) * ' ' + ' ').join(chunks)
    tqdm.write(msg)
