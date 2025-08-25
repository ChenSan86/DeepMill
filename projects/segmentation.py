# --------------------------------------------------------  # 文件头，版权声明和作者信息
# Octree-based Sparse Convolutional Neural Networks         # 项目名称
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>   # 版权信息
# Licensed under The MIT License [see LICENSE for details]   # 许可证信息
# Written by Peng-Shuai Wang                                # 作者信息
# --------------------------------------------------------

import os  # 导入os模块，进行文件和路径操作
import torch  # 导入PyTorch库
import ocnn  # 导入ocnn库，包含点云相关模型和工具
import numpy as np  # 导入numpy库，进行数值计算
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from thsolver import Solver  # 导入自定义Solver基类

from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)  # 导入数据集构建函数
import pdb  # 导入pdb调试工具
from sklearn.metrics import f1_score  # 导入F1分数计算函数
# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
# 解决PyTorch多进程共享内存的兼容性问题
torch.multiprocessing.set_sharing_strategy('file_system')


class SegSolver(Solver):  # 继承自Solver，分割任务专用训练器

    def get_model(self, flags):  # 根据配置返回分割模型
        if flags.name.lower() == 'segnet':  # 如果模型名为segnet
            model = ocnn.models.SegNet(
                flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)  # 构建SegNet模型
#TODO ======================================================================================
        elif flags.name.lower() == 'unet':  # 如果模型名为unet
            model = ocnn.models.UNet(
                flags.channel, flags.nout, flags.interp, flags.nempty)  # 构建UNet模型
        else:
            raise ValueError  # 未知模型名抛出异常
        return model  # 返回模型对象
#TODO ======================================================================================
    def get_dataset(self, flags):  # 根据配置返回数据集和collate函数
        #TODO ==================================================================================
        if flags.name.lower() == 'shapenet':  # ShapeNet分割数据集
            return get_seg_shapenet_dataset(flags)
        elif flags.name.lower() == 'scannet':  # ScanNet分割数据集
            return get_scannet_dataset(flags)
        elif flags.name.lower() == 'kitti':  # KITTI分割数据集
            return get_kitti_dataset(flags)
        else:
            raise ValueError  # 未知数据集名抛出异常

    def get_input_feature(self, octree):  # 获取输入特征（待实现）
        flags = self.FLAGS.MODEL  # 获取模型相关配置
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)  # 输入特征提取模块
        data = octree_feature(octree)  # 提取特征
        return data  # 返回特征

    def process_batch(self, batch, flags):  # 处理一个batch的数据
        def points2octree(points):  # 点云转八叉树
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)  # 创建八叉树对象
            octree.build_octree(points)  # 构建八叉树
            return octree  # 返回八叉树对象

        if 'octree' in batch:  # 如果batch中已经有octree
            batch['octree'] = batch['octree'].cuda(non_blocking=True)  # 将octree移到GPU
            batch['points'] = batch['points'].cuda(non_blocking=True)  # 将点云移到GPU
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        else:  # 如果batch中没有octree
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]  # 将点云移到GPU
            octrees = [points2octree(pts) for pts in points]  # 将点云转换为八叉树
            octree = ocnn.octree.merge_octrees(octrees)  # 合并多个八叉树
            octree.construct_all_neigh()  # 构建所有邻居关系
            batch['points'] = ocnn.octree.merge_points(points)  # 合并点云
            batch['octree'] = octree  # 将八叉树添加到batch中
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        return batch  # 返回处理后的batch


    def model_forward(self, batch):  # 模型前向传播
        octree, points = batch['octree'], batch['points']  # 获取octree和points
        data = self.get_input_feature(octree)  # 获取输入特征
        query_pts = torch.cat([points.points, points.batch_id], dim=1)  # 拼接点云坐标和batch_id

        # 从 batch 中提取刀具参数
        tool_params = batch['tool_params']  # 获取刀具参数
        # print(f"Original tool_params: {tool_params}, type: {type(tool_params)}")
        tool_params = [[float(item) for item in row] for row in tool_params]  # 转换为浮点数
        tool_params = torch.tensor(tool_params, dtype=torch.float32).cuda() #FC: 需要标注GPU序号
        # print(f"Processed tool_params: {tool_params}, type: {type(tool_params)}, shape: {tool_params.shape}")

        # 将刀具参数传递给模型
        logit_1,logit_2 = self.model.forward(data, octree, octree.depth, query_pts, tool_params)  # 传递刀具参数
        labels = points.labels.squeeze(1)  # 获取标签
        label_mask = labels > self.FLAGS.LOSS.mask  # 过滤标签
        labels_2 = points.labels_2.squeeze(1)  # 获取第二组标签
        return logit_1[label_mask], logit_2[label_mask], labels[label_mask], labels_2[label_mask]  # 返回有效的logit和标签


    def visualization(self, points, logit, labels,  red_folder,gt_folder):  # 可视化函数
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:  # 打开红色点云文件
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] > logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")  # 写入点坐标

        with open(gt_folder, 'w') as obj_file:  # 打开绿色点云文件
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 0:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")  # 写入点坐标

    def visualization1(self, points, logit, labels,  red_folder,gt_folder):  # 可视化函数
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:  # 打开红色点云文件
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] < logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")  # 写入点坐标

        with open(gt_folder, 'w') as obj_file:  # 打开绿色点云文件
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 1:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")  # 写入点坐标


    def train_step(self, batch):  # 训练步骤
        batch = self.process_batch(batch, self.FLAGS.DATA.train)  # 处理训练数据
        logit_1,logit_2, label, label_2 = self.model_forward(batch)  # 前向传播
        #TODO loss使用6->3*3 后使用L2矩阵差平方和（Frobenius norm 的平方）
        loss_1 = self.loss_function(logit_1, label)  # 计算损失
        loss_2 = self.loss_function(logit_2, label_2)  # 计算损失
        loss = (loss_1 + loss_2)/2  # 平均损失
        accu_1 = self.accuracy(logit_1, label)  # 计算准确率
        accu_2 = self.accuracy(logit_2, label_2)  # 计算准确率
        accu = (accu_1 + accu_2)/2  # 平均准确率

        pred_1 = logit_1.argmax(dim=-1)  # 假设 logit_1 是 logits 形式，需要用 argmax 选取预测类别
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        #TODO 测地距离（geodesic error） 衡量旋转误差 ；平均误差、最大误差、标准差，并画出误差分布百分位曲线。
        f1_score_1 = f1_score(label.cpu().numpy(), pred_1.cpu().numpy(), average='binary')  # 计算F1分数
        f1_score_2 = f1_score(label_2.cpu().numpy(), pred_2.cpu().numpy(), average='binary')  # 计算F1分数
        f1_score_avg = (f1_score_1 + f1_score_2) / 2  # 平均F1分数

        return {'train/loss': loss, 'train/accu': accu, 'train/accu_red': accu_1, 'train/accu_green': accu_2,
                'train/f1_red': torch.tensor(f1_score_1, dtype=torch.float32).cuda(), 'train/f1_green': torch.tensor(f1_score_2, dtype=torch.float32).cuda(), 'train/f1_avg': torch.tensor(f1_score_avg, dtype=torch.float32).cuda()}
        # return {'train/loss': loss, 'train/accu': accu,'train/accu_red': accu_1,'train/accu_green': accu_2,
        # 'train/f1_red': f1_score_1,'train/f1_green': f1_score_2,'train/f1_avg': f1_score_avg}



    def test_step(self, batch):  # 测试步骤
        batch = self.process_batch(batch, self.FLAGS.DATA.test)  # 处理测试数据
        with torch.no_grad():
            logit_1,logit_2, label, label_2 = self.model_forward(batch)  # 前向传播
        # self.visualization(batch['points'], logit, label, ".\\data\\vis\\"+batch['filename'][0][:-4]+".obj") #FC:目前可视化只支持test的batch size=1
        loss_1 = self.loss_function(logit_1, label)  # 计算损失
        loss_2 = self.loss_function(logit_2, label_2)  # 计算损失
        loss = (loss_1 + loss_2) / 2  # 平均损失
        accu_1 = self.accuracy(logit_1, label)  # 计算准确率
        accu_2 = self.accuracy(logit_2, label_2)  # 计算准确率
        accu = (accu_1 + accu_2) / 2  # 平均准确率
        num_class = self.FLAGS.LOSS.num_class  # 获取类别数量
        IoU, insc, union = self.IoU_per_shape(logit_1, label, num_class)  # 计算每个形状的IoU

        folders = [
            './visual/red_points',
            './visual/GT_red',
            './visual/green_points',
            './visual/GT_green'
        ]
        for folder in folders:  # 创建可视化结果保存文件夹
            if not os.path.exists(folder):
                os.makedirs(folder)
              
        red_folder = os.path.join(r"./visual/red_points",
                                  batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                      0] + ".obj")  # 红色点云文件路径
        gt_red_folder = os.path.join(r"./visual/GT_red",
                                     batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                         0] + ".obj")  # 红色点云GT文件路径
        green_folder = os.path.join(r'./visual/green_points',
                                    batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                        0] + ".obj")  # 绿色点云文件路径
        gt_green_folder = os.path.join(r'./visual/GT_green',
                                       batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                           0] + ".obj")  # 绿色点云GT文件路径
        self.visualization(batch['points'], logit_1, label, red_folder, gt_red_folder)  # 可视化红色点云
        self.visualization1(batch['points'], logit_2, label_2, green_folder, gt_green_folder)  # 可视化绿色点云
        pred_1 = logit_1.argmax(dim=-1)  # 假设 logit_1 是 logits 形式，需要用 argmax 选取预测类别
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(label.cpu().numpy(), pred_1.cpu().numpy(), average='binary')  # 计算F1分数
        f1_score_2 = f1_score(label_2.cpu().numpy(), pred_2.cpu().numpy(), average='binary')  # 计算F1分数
        f1_score_avg = (f1_score_1 + f1_score_2) / 2  # 平均F1分数

        names = ['test/loss', 'test/accu', 'test/accu_red','test/accu_green','test/mIoU', 'test/f1_red','test/f1_green','test/f1_avg'] + \
                ['test/intsc_%d' % i for i in range(num_class)] + \
                ['test/union_%d' % i for i in range(num_class)]
        tensors = [loss, accu, accu_1, accu_2, IoU, torch.tensor(f1_score_1, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_2, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_avg, dtype=torch.float32).cuda()] + insc + union
        return dict(zip(names, tensors))  # 返回测试结果


    def eval_step(self, batch):  # 评估步骤
        batch = self.process_batch(batch, self.FLAGS.DATA.test)  # 处理评估数据
        with torch.no_grad():
            logit, _ = self.model_forward(batch)  # 前向传播
        prob = torch.nn.functional.softmax(logit, dim=1)  # 计算类别概率

        # split predictions
        inbox_masks = batch['inbox_mask']  # 获取边界框掩码
        npts = batch['points'].batch_npt.tolist()  # 获取每个点云的点数
        probs = torch.split(prob, npts)  # 按照点数拆分概率

        # merge predictions
        batch_size = len(inbox_masks)  # 批次大小
        for i in range(batch_size):
            # The point cloud may be clipped when doing data augmentation. The
            # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
            # contains the prediction for all points.
            prob = probs[i].cpu()  # 获取CPU上的概率
            inbox_mask = inbox_masks[i].to(prob.device)  # 获取掩码
            prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])  # 创建全零概率张量
            prob_all_pts[inbox_mask] = prob  # 填充未裁剪点的概率

            # Aggregate predictions across different epochs
            filename = batch['filename'][i]  # 获取文件名
            self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts  # 累加概率

            # Save the prediction results in the last epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')  # 结果保存路径
                curr_folder = os.path.dirname(full_filename)  # 获取文件夹路径
                if not os.path.exists(curr_folder): os.makedirs(curr_folder)  # 创建文件夹
                np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())  # 保存结果

    def result_callback(self, avg_tracker, epoch):  # 结果回调函数
        r''' Calculate the part mIoU for PartNet and ScanNet.
        '''

        iou_part = 0.0
        avg = avg_tracker.average()  # 获取平均值

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # PartNet are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1  # 获取掩码
        num_class = self.FLAGS.LOSS.num_class  # 获取类别数量
        for i in range(mask, num_class):
            instc_i = avg['test/intsc_%d' % i]  # 获取交集
            union_i = avg['test/union_%d' % i]  # 获取并集
            iou_part += instc_i / (union_i + 1.0e-10)  # 计算IoU

        iou_part = iou_part / (num_class - mask)  # 平均IoU

        avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})  # 更新Tracker
        tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))  # 打印信息

    def loss_function(self, logit, label):  # 损失函数
        """
        计算交叉熵损失函数。
        参数：
            logit: 网络输出的未归一化分数（shape: [N, num_class]）
            label: 真实标签（shape: [N]），需为整数类型
        返回：
            loss: 标量，交叉熵损失
        """
        criterion = torch.nn.CrossEntropyLoss()  # 创建交叉熵损失函数
        loss = criterion(logit, label.long())  # 计算损失，标签需为long类型
        return loss  # 返回损失

    def accuracy(self, logit, label):  # 准确率计算
        """
        计算分类准确率。
        参数：
            logit: 网络输出的未归一化分数（shape: [N, num_class]）
            label: 真实标签（shape: [N]）
        返回：
            accu: 标量，准确率（0~1之间）
        """
        pred = logit.argmax(dim=1)  # 取最大分数作为预测类别
        accu = pred.eq(label).float().mean()  # 计算预测与真实标签相��的比例
        return accu  # 返回准确率

    def IoU_per_shape(self, logit, label, class_num):  # 计算每个形状的IoU
        """
        计算单个样本的每类IoU（交并比），并返回平均IoU。
        参数：
            logit: 网络输出的未归一化分数（shape: [N, num_class]）
            label: 真实标签（shape: [N]）
            class_num: 类别总数
        返回：
            IoU: 平均IoU（标量）
            intsc: 每类交集数量列表
            union: 每类并集数量列表
        """
        pred = logit.argmax(dim=1)  # 取最大分数作���预测类别

        IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10  # 初始化IoU、有效类别数、极小值防止除零
        intsc, union = [None] * class_num, [None] * class_num  # 初始化交集和并集列表
        for k in range(class_num):  # 遍历每个类别
            pk, lk = pred.eq(k), label.eq(k)  # 预测为k和真实为k的布尔掩码
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())  # 交集数量
            union[k] = torch.sum(torch.logical_or(pk, lk).float())  # 并集数量

            valid = torch.sum(lk.any()) > 0  # 判断该类别是否在标签中出现
            valid_part_num += valid.item()  # 有效类别计数
            IoU += valid * intsc[k] / (union[k] + esp)  # 累加有效类别的IoU

        # 对ShapeNet，平均IoU按有效类别数归一化
        IoU /= valid_part_num + esp  # 防止除零
        return IoU, intsc, union  # 返回平均IoU、交集、并集


if __name__ == "__main__":

    SegSolver.main()
