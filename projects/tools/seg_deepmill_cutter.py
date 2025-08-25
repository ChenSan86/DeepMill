# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os  # 导入操作系统相关的库，用于路径处理、文件操作等
import json  # 导入json库，用于处理json格式数据
import argparse  # 导入命令行参数解析库，用于解析命令行输入
import wget  # 导入wget库，用于下载文件
import zipfile  # 导入zipfile库，用于解压zip文件
import ssl  # 导入ssl库，用于处理安全连接
import numpy as np  # 导入numpy库，用于数值计算和数组操作
import shutil  # 导入shutil库，用于文件和文件夹的高级操作
import utils  # 导入自定义工具模块

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加参数 --run，指定要运行的命令，默认值为'prepare_dataset'
parser.add_argument('--run', type=str, required=False, default='prepare_dataset',
                    help='The command to run.')
# 添加参数 --sr，指定训练和验证数据的划分比例，默认值为0.8
parser.add_argument('--sr', type=float, required=False, default=0.8,
                    help='tran and valid data split ration.')
# 解析命令行参数
args = parser.parse_args()

# 下面这行用于解决使用wget时出现的"SSL: CERTIFICATE_VERIFY_FAILED"错误
# 参考：https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前文件的上一级目录的绝对路径
abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 定义数据根目录
root_folder = os.path.join(abs_path, 'data')#data/
# 定义原始数据压缩包名称
zip_name = 'raw_data'
# 定义原始数据txt文件夹路径
txt_folder = os.path.join(root_folder, zip_name)#data/raw_data
# 定义点云数据文件夹路径
ply_folder = os.path.join(root_folder, 'points')#data/raw_data/points

# 定义类别列表
categories = ['models']
# 定义名称列表
names = ['models']
# 定义分割数量列表
seg_num = [2]
# 定义距离阈值列表
# dis = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]


def normalize_points(input_folder, output_folder):
    """对点云数据进行归一化处理
    Parameters:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        # 只处理以.txt结尾的文件
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)  # 输入文件完整路径
            output_path = os.path.join(output_folder, filename)  # 输出文件完整路径

            with open(input_path, 'r') as f:
                lines = f.readlines()  # 读取文件所有行

            data = []
            # 遍历每一行数据
            for line in lines:
                # 如果行不为空且不以'#'开头
                if line.strip() and not line.strip().startswith('#'):
                    data_data = [x for x in line.split()]  # 按空格分割数据
                    # 如果数据中包含"-nan(ind)"，则将其替换为[0, 0, 1]
                    if "-nan(ind)" in data_data[3:6]:
                        data_data[3:6] = ["0","0","1"]
                    data.append(data_data)  # 将处理后的数据添加到列表中

            data = np.array(data)  # 转换为numpy数组
            data = data.astype(float)  # 转换数据类型为float
            xyz = data[:, :3]  # 提取xyz坐标
            rest = data[:, 3:]  # 提取其余数据

            center = np.mean(xyz, axis=0)  # 计算中心点坐标
            xyz_centered = xyz - center  # 坐标去中心化

            max_extent = np.max(np.abs(xyz_centered), axis=0)  # 计算最大扩展范围
            max_scale = np.max(max_extent)  # 计算最大缩放比例

            # 归一化坐标到[-0.8, 0.8]范围
            xyz_normalized = xyz_centered / max_scale * 0.8
            normalized_data = np.hstack((xyz_normalized, rest))  # 合并归一化后的坐标和其余数据
            np.savetxt(output_path, normalized_data, fmt="%.6f")  # 保存处理后的数据

            print(f"Processed and saved: {filename}")  # 打印处理完成的文件名



def txt_to_ply():
  """将txt文件转换为ply文件"""
  print('-> Convert txt files to ply files')
  # 遍历所有类别
  for i, c in enumerate(categories):
    src_folder = os.path.join(txt_folder, c)  # 源文件夹路径
    des_folder = os.path.join(ply_folder, c)  # 目标文件夹路径
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    filenames = os.listdir(src_folder)  # 获取源文件夹中的所有文件名
    # 遍历每个文件
    for filename in filenames:
      filename_txt = os.path.join(src_folder, filename)  # txt文件完整路径
      filename_ply = os.path.join(des_folder, filename[:-4] + '.ply')  # ply文件完整路径

      raw = np.loadtxt(filename_txt)  # 读取txt文件数据
      points = raw[:, :3]  # 提取点坐标
      normals = raw[:, 3:6]  # 提取法线数据
      label = raw[:, 6:7]   # 提取标签数据，注意这里的标签是位移信息
      label_2 = raw[:, 7:]  # 提取其余标签数据

      # 保存为ply文件
      utils.save_points_to_ply(
          filename_ply, points, normals, labels=label, labels_2=label_2, text=False)
      print('Save: ' + os.path.basename(filename_ply))  # 打印保存的文件名


def generate_filelist():
  """生成训练和测试文件列表"""
  print('-> Generate filelists')
  list_folder = os.path.join(root_folder, 'filelist')  # 文件列表存放文件夹
  # 如果文件列表文件夹不存在，则创建
  if not os.path.exists(list_folder):
    os.makedirs(list_folder)
  ratio = args.sr  # 获取训练和验证数据的划分比例

  # 遍历所有类别
  for i, c in enumerate(categories):

    all_file = os.listdir(os.path.join(txt_folder, c))  # 获取当前类别下的所有文件
    train_val_filelist = []  # 初始化训练和验证文件列表
    test_filelist = []  # 初始化测试文件列表

    # 遍历每个文件
    for filename in all_file:
      ply_filename = os.path.join(c, filename[:-4] + '.ply')  # 对应的ply文件名

      cutter_filename = os.path.join(txt_folder, c+'_cutter', filename[:-4] + '_cutter.txt')  # 切割文件名

      four_numbers = []  # 初始化四个数字列表
      # 如果切割文件存在
      if os.path.exists(cutter_filename):
        cutter_data = np.loadtxt(cutter_filename)  # 读取切割文件数据
        four_numbers = cutter_data[:4]  # 获取前四个数字

      # 文件条目格式：<ply文件名> <类别索引> <四个数字>
      #TODO 在这里加入标签<ply文件名> <类别索引> <四个数字> <六个数字>
      file_entry = '%s %d %.6f %.6f %.6f %.6f' % (ply_filename, i, *four_numbers)

      # 按照比例将文件分配到训练和验证集或测试集中
      if len(train_val_filelist) < int(ratio * len(all_file)):
        train_val_filelist.append(file_entry)  # 添加到训练和验证文件列表
      else:
        test_filelist.append(file_entry)  # 添加到测试文件列表

    # 保存训练和验证文件列表
    filelist_name = os.path.join(list_folder, c + '_train_val.txt')
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(train_val_filelist))

    # 保存测试文件列表
    filelist_name = os.path.join(list_folder, c + '_test.txt')
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(test_filelist))


def prepare_dataset():
    """准备数据集，包括归一化点云数据、转换文件格式和生成文件列表"""
    for i, c in enumerate(categories):
        input_folder = os.path.join(txt_folder, c)  # 输入文件夹路径
        normalize_points(input_folder, input_folder)  # 归一化点云数据
    txt_to_ply()  # 转换文件格式
    generate_filelist()  # 生成文件列表


# 程序入口
if __name__ == '__main__':
    eval('%s()' % args.run)  # 根据命令行参数运行相应的函数
