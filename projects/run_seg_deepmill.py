# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os  # 导入操作系统相关库，用于路径处理和文件操作
import math  # 导入数学库，用于数学运算
import argparse  # 导入命令行参数解析库
import numpy as np  # 导入numpy库，用于数值计算
import pdb  # 导入pdb库，用于调试
import subprocess  # 导入子进程库，用于执行命令

parser = argparse.ArgumentParser()  # 创建命令行参数解析器
parser.add_argument('--alias', type=str, default='unet_d5')  # 添加参数，训练日志别名
parser.add_argument('--gpu', type=str, default='0')  # 添加参数，使用的GPU编号
parser.add_argument('--depth', type=int, default=5)  # 添加参数，网络深度
parser.add_argument('--model', type=str, default='unet')  # 添加参数，模型类型
parser.add_argument('--mode', type=str, default='randinit')  # 添加参数，初始化模式
parser.add_argument('--ckpt', type=str, default='\'\'')  # 添加参数，权重路径
parser.add_argument('--ratios', type=float, default=[1], nargs='*')  # 添加参数，数据比例列表

args = parser.parse_args()  # 解析命令行参数
alias = args.alias  # 获取训练日志别名
gpu = args.gpu  # 获取GPU编号
mode = args.mode  # 获取初始化模式
ratios = args.ratios  # 获取数据比例列表
# ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]  # 可选数据比例

module = 'segmentation.py'  # 主训练脚本名
script = 'python %s --config configs/seg_deepmill.yaml' % module  # 构建训练命令

data = 'data'  # 数据目录
logdir = 'logs/seg_deepmill'  # 日志目录

categories = ['models']  # 类别列表
names = ['models']  # 名称列表
seg_num = [2]  # 分割类别数
train_num = [4471]  # 训练样本数
test_num = [1118]  # 测试样本数
max_epoches = [1500]  # 最大训练轮数
max_iters = [1500]  # 最大迭代次数

for i in range(len(ratios)):  # 遍历所有数据比例
    for k in range(len(categories)):  # 遍历所有类别
        ratio, cat = ratios[i], categories[k]  # 当前比例和类别
        mul = 2 if ratios[i] < 0.1 else 1  # 数据比例小于0.1时训练轮数加倍
        max_epoch = int(max_epoches[k] * ratio * mul)  # 计算最大训练轮数
        milestone1, milestone2 = int(0.5 * max_epoch), int(0.25 * max_epoch)  # 学习率里程碑
        # test_every_epoch = int(math.ceil(max_epoch * 0.02))  # 测试间隔
        test_every_epoch = 50  # 固定每50轮测试一次
        take = int(math.ceil(train_num[k] * ratio))  # 实际训练样本数
        logs = os.path.join(
            logdir, '{}/{}_{}/ratio_{:.2f}'.format(alias, cat, names[k], ratio))  # 日志目录

        cmds = [  # 构建训练命令参数列表
            script,
            'SOLVER.gpu {},'.format(gpu),
            'SOLVER.logdir {}'.format(logs),
            'SOLVER.max_epoch {}'.format(max_epoch),
            'SOLVER.milestones {},{}'.format(milestone1, milestone2),
            'SOLVER.test_every_epoch {}'.format(test_every_epoch),
            'SOLVER.ckpt {}'.format(args.ckpt),
            'DATA.train.depth {}'.format(args.depth),
            'DATA.train.filelist {}/filelist/{}_train_val.txt'.format(data, cat),
            'DATA.train.take {}'.format(take),
            'DATA.test.depth {}'.format(args.depth),
            'DATA.test.filelist {}/filelist/{}_test.txt'.format(data, cat),
            'MODEL.stages {}'.format(args.depth - 2),
            'MODEL.nout {}'.format(seg_num[k]),
            'MODEL.name {}'.format(args.model),
            'LOSS.num_class {}'.format(seg_num[k])
        ]

        cmd = ' '.join(cmds)  # 拼接命令为字符串
        print('\n', cmd, '\n')  # 打印命令
        # os.system(cmd)  # 可选：用os.system执行命令
        subprocess.run(cmd)  # 用subprocess执行命令

summary = []  # 汇总结果列表
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')  # 添加类别名
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))  # 添加训练样本数
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))  # 添加测试样本数

for i in range(len(ratios)-1, -1, -1):  # 逆序遍历所有数据比例
    ious = [None] * len(categories)  # 初始化IoU列表
    for j in range(len(categories)):  # 遍历所有类别
        filename = '{}/{}/{}_{}/ratio_{:.2f}/log.csv'.format(
            logdir, alias, categories[j], names[j], ratios[i])  # 构建日志文件路径
        with open(filename, newline='') as fid:  # 打开日志文件
            lines = fid.readlines()  # 读取所有行
        last_line = lines[-1]  # 获取最后一行
        pos = last_line.find('test/mIoU:')  # 查找mIoU位置
        ious[j] = float(last_line[pos+11:pos+16])  # 解析IoU数值
    CmIoU = np.array(ious).mean()  # 计算类别平均mIoU
    ImIoU = np.sum(np.array(ious)*np.array(test_num)) / np.sum(np.array(test_num))  # 计算实例平均mIoU
    ious = [str(iou) for iou in ious] + \
           ['{:.3f}'.format(CmIoU), '{:.3f}'.format(ImIoU)]  # 拼接IoU结果
    summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))  # 添加到汇总列表

with open('{}/{}/summaries.csv'.format(logdir, alias), 'w') as fid:  # 打开汇总文件
    summ = '\n'.join(summary)  # 拼���所有汇总结果
    fid.write(summ)  # 写入文件
    print(summ)  # 打印汇总结果
