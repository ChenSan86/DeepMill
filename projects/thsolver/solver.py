# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os  # 操作系统相关库，用于路径、文件操作
import torch  # PyTorch深度学习库
import torch.nn  # 神经网络相关模块
import torch.optim  # 优化器相关模块
import torch.distributed  # 分布式训练相关模块
import torch.multiprocessing  # 多进程相关模块
import torch.utils.data  # 数据加载相关模块
import random  # 随机数生成
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
from packaging import version  # 版本管理
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志

from .sampler import InfSampler, DistributedInfSampler  # 无限采样器
from .tracker import AverageTracker  # 平均值追踪器
from .config import parse_args  # 配置参数解析
from .lr_scheduler import get_lr_scheduler  # 学习率调度器


class Solver:
    """
    通用训练器基类，支持单卡/多卡训练、分布式训练、模型保存/加载、日志记录等。
    需子类实现 get_model、get_dataset、train_step、test_step、eval_step。
    """

    def __init__(self, FLAGS, is_master=True):
        self.FLAGS = FLAGS  # 配置参数
        self.is_master = is_master  # 是否为主进程
        self.world_size = len(FLAGS.SOLVER.gpu)  # GPU数量
        self.device = torch.cuda.current_device()  # 当前GPU设备
        self.disable_tqdm = not (is_master and FLAGS.SOLVER.progress_bar)  # 是否禁用进度条
        self.start_epoch = 1  # 起始epoch

        # 训练相关对象
        self.model = None           # 模型对象
        self.optimizer = None       # 优化器对象
        self.scheduler = None       # 学习率调度器
        self.summary_writer = None  # TensorBoard日志
        self.log_file = None        # 日志文件路径
        self.eval_rst = dict()      # 评估结果字典
        self.best_val = None        # 最优验证结果

    def get_model(self):
        '''返回模型对象，需子类实现'''
        raise NotImplementedError

    def get_dataset(self, flags):
        '''返回数据集和collate函数，需子类实现'''
        raise NotImplementedError

    def train_step(self, batch):
        '''返回训练损失和信息，需包含'train/loss'键，子类实现'''
        raise NotImplementedError

    def test_step(self, batch):
        '''返回测试损失和信息，子类实现'''
        raise NotImplementedError

    def eval_step(self, batch):
        '''评估模型，子类实现'''
        raise NotImplementedError

    def result_callback(self, avg_tracker: AverageTracker, epoch):
        '''可选：根据平均值追踪器做额外操作'''
        pass

    def config_dataloader(self, disable_train_data=False):
        '''配置训练和测试数据加载器'''
        flags_train, flags_test = self.FLAGS.DATA.train, self.FLAGS.DATA.test
        if not disable_train_data and not flags_train.disable:
            self.train_loader = self.get_dataloader(flags_train)
            self.train_iter = iter(self.train_loader)
        if not flags_test.disable:
            self.test_loader = self.get_dataloader(flags_test)
            self.test_iter = iter(self.test_loader)

    def get_dataloader(self, flags):
        '''根据配置创建DataLoader，支持分布式采样'''
        dataset, collate_fn = self.get_dataset(flags)
        if self.world_size > 1:
            sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
        else:
            sampler = InfSampler(dataset, shuffle=flags.shuffle)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
            sampler=sampler, collate_fn=collate_fn, pin_memory=flags.pin_memory)
        return data_loader

    def config_model(self):
        '''配置模型，支持SyncBN和DDP分布式训练'''
        flags = self.FLAGS.MODEL
        model = self.get_model(flags)
        model.cuda(device=self.device)
        if self.world_size > 1:
            if flags.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[self.device],
                output_device=self.device, broadcast_buffers=False,
                find_unused_parameters=flags.find_unused_parameters)
        if self.is_master:
            print(model)  # 打印模型结构
            total_params = sum(p.numel() for p in model.parameters())
            print("Total number of parameters: %.3fM" % (total_params / 1e6))
        self.model = model

    def config_optimizer(self):
        '''配���优化器，支持SGD/Adam/AdamW'''
        flags = self.FLAGS.SOLVER
        base_lr = flags.lr * self.world_size
        parameters = self.model.parameters()
        if flags.type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                parameters, lr=base_lr, weight_decay=flags.weight_decay, momentum=0.9)
        elif flags.type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                parameters, lr=base_lr, weight_decay=flags.weight_decay)
        elif flags.type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                parameters, lr=base_lr, weight_decay=flags.weight_decay)
        else:
            raise ValueError

    def config_lr_scheduler(self):
        '''配置学习率调度器'''
        self.scheduler = get_lr_scheduler(self.optimizer, self.FLAGS.SOLVER)

    def configure_log(self, set_writer=True):
        '''配置日志和TensorBoard写入器'''
        self.logdir = self.FLAGS.SOLVER.logdir
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.log_file = os.path.join(self.logdir, 'log.csv')
        if self.is_master:
            tqdm.write('Logdir: ' + self.logdir)
        if self.is_master and set_writer:
            self.summary_writer = SummaryWriter(self.logdir, flush_secs=20)
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_epoch(self, epoch):
        '''单个epoch的训练流程，包含梯度更新、日志记录、分布式同步等'''
        self.model.train()
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        train_tracker = AverageTracker()
        rng = range(len(self.train_loader))
        log_per_iter = self.FLAGS.SOLVER.log_per_iter
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            # 每50步清理一次显存缓存
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()
            batch = next(self.train_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch
            self.optimizer.zero_grad()
            output = self.train_step(batch)
            output['train/loss'].backward()
            # 梯度裁剪
            clip_grad = self.FLAGS.SOLVER.clip_grad
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()
            train_tracker.update(output)
            # 输出中间日志
            if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
                notes = 'iter: %d' % it
                train_tracker.log(epoch, msg_tag='- ', notes=notes, print_time=False)
        # 保存日志
        if self.world_size > 1:
            train_tracker.average_all_gather()
        if self.is_master:
            train_tracker.log(epoch, self.summary_writer)

    def test_epoch(self, epoch):
        '''单个epoch的测试流程，包含日志记录、分布式同步等'''
        self.model.eval()
        test_tracker = AverageTracker()
        rng = range(len(self.test_loader))
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()
            batch = next(self.test_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch
            output = self.test_step(batch)
            test_tracker.update(output, record_time=False)
        if self.world_size > 1:
            test_tracker.average_all_gather()
        if self.is_master:
            self.result_callback(test_tracker, epoch)
            self.save_best_checkpoint(test_tracker, epoch)
            test_tracker.log(epoch, self.summary_writer, self.log_file, msg_tag='=>')

    def eval_epoch(self, epoch):
        '''单个epoch的评估流程，支持自定义评估步数'''
        self.model.eval()
        eval_step = min(self.FLAGS.SOLVER.eval_step, len(self.test_loader))
        if eval_step < 1:
            eval_step = len(self.test_loader)
        for it in tqdm(range(eval_step), ncols=80, leave=False):
            batch = next(self.test_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch
            with torch.no_grad():
                self.eval_step(batch)

    def save_best_checkpoint(self, tracker: AverageTracker, epoch: int):
        '''保存最优模型权重，支持max/min模式自动选择'''
        best_val = self.FLAGS.SOLVER.best_val
        if not (best_val and self.FLAGS.SOLVER.run == 'train'):
            return
        compare, key = best_val.split(':')
        key = 'test/' + key
        assert compare in ['max', 'min']
        operator = lambda x, y: x > y if compare == 'max' else x < y
        if key in tracker.value:
            curr_val = (tracker.value[key] / tracker.num[key]).item()
            if self.best_val is None or operator(curr_val, self.best_val):
                self.best_val = curr_val
                model_dict = (self.model.module.state_dict() if self.world_size > 1
                              else self.model.state_dict())
                torch.save(model_dict, os.path.join(self.logdir, 'best_model.pth'))
                msg = 'epoch: %d, %s: %f' % (epoch, key, curr_val)
                with open(os.path.join(self.logdir, 'best_model.txt'), 'a') as fid:
                    fid.write(msg + '\n')
                tqdm.write('=> Best model at ' + msg)

    def save_checkpoint(self, epoch):
        '''保存当前epoch的模型和优化器状态，自动清理旧ckpt'''
        if not self.is_master: return
        # clean up
        ckpts = sorted(os.listdir(self.ckpt_dir))
        ckpts = [ck for ck in ckpts if ck.endswith('.pth') or ck.endswith('.tar')]
        if len(ckpts) > self.FLAGS.SOLVER.ckpt_num:
            for ckpt in ckpts[:-self.FLAGS.SOLVER.ckpt_num]:
                os.remove(os.path.join(self.ckpt_dir, ckpt))
        # save ckpt
        model_dict = (self.model.module.state_dict() if self.world_size > 1
                      else self.model.state_dict())
        ckpt_name = os.path.join(self.ckpt_dir, '%05d' % epoch)
        torch.save(model_dict, ckpt_name + '.model.pth')
        torch.save({'model_dict': model_dict, 'epoch': epoch,
                    'optimizer_dict': self.optimizer.state_dict(),
                    'scheduler_dict': self.scheduler.state_dict(), },
                   ckpt_name + '.solver.tar')

    def load_checkpoint(self):
        '''加载模型和优化器状态，支持自动查找最新ckpt'''
        ckpt = self.FLAGS.SOLVER.ckpt
        if not ckpt:
            # If ckpt is empty, then get the latest checkpoint from ckpt_dir
            if not os.path.exists(self.ckpt_dir):
                return
            ckpts = sorted(os.listdir(self.ckpt_dir))
            ckpts = [ck for ck in ckpts if ck.endswith('solver.tar')]
            if len(ckpts) > 0:
                ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
        if not ckpt:
            return  # return if ckpt is still empty

        # load trained model
        # check: map_location = {'cuda:0' : 'cuda:%d' % self.rank}
        trained_dict = torch.load(ckpt, map_location='cuda')
        if ckpt.endswith('.solver.tar'):
            model_dict = trained_dict['model_dict']
            self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
            if self.optimizer:
                self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
        else:
            model_dict = trained_dict
        model = self.model.module if self.world_size > 1 else self.model
        model.load_state_dict(model_dict)

        # print messages
        if self.is_master:
            tqdm.write('Load the checkpoint: %s' % ckpt)
            tqdm.write('The start_epoch is %d' % self.start_epoch)

    def manual_seed(self):
        '''手动设置随机种子，确保可重复性'''
        rand_seed = self.FLAGS.SOLVER.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def train(self):
        '''训练主流程，包含模型、数据加载器、优化器、学习率调度器配置，以及训练循环'''
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_lr_scheduler()
        self.configure_log()
        self.load_checkpoint()

        rng = range(self.start_epoch, self.FLAGS.SOLVER.max_epoch+1)
        for epoch in tqdm(rng, ncols=80, disable=self.disable_tqdm):
            # training epoch
            self.train_epoch(epoch)

            # update learning rate
            self.scheduler.step()
            if self.is_master:
                lr = self.scheduler.get_last_lr()  # lr is a list
                self.summary_writer.add_scalar('train/lr', lr[0], epoch)

            # testing or not
            if epoch % self.FLAGS.SOLVER.test_every_epoch != 0:
                continue

            # testing epoch
            self.test_epoch(epoch)

            # checkpoint
            self.save_checkpoint(epoch)

        # sync and exit
        if self.world_size > 1:
            torch.distributed.barrier()

    def test(self):
        '''测试主流程，仅包含模型、数据加载器配置，以及测试循环'''
        self.manual_seed()  # 设置随机种子，保证结果可复现
        self.config_model()  # 配置模型（加载结构、参数等）
        self.configure_log(set_writer=False)  # 配置日志系统，不写入TensorBoard
        self.config_dataloader(disable_train_data=True)  # 配置数据加载器，仅加载测试集
        self.load_checkpoint()  # 加载模型权重和优化器状态
        self.test_epoch(epoch=0)  # 执行测试流程，epoch设为0

    def evaluate(self):
        '''评估主流程，仅包含模型、数据加载器配置，以及评估循环'''
        self.manual_seed()  # 设置随机种子
        self.config_model()  # 配置模型
        self.configure_log(set_writer=False)  # 配置日志系统
        self.config_dataloader(disable_train_data=True)  # 配置数据加载器，仅加载测试集
        self.load_checkpoint()  # 加载模型权重
        for epoch in tqdm(range(self.FLAGS.SOLVER.eval_epoch), ncols=80):  # 按配置循环评估次数
            self.eval_epoch(epoch)  # 执行单次评估流程

    def profile(self):
        r''' 设置`DATA.train.num_workers 0`后使用此函数。 '''
        self.config_model()  # 配置模型
        self.config_dataloader()  # 配置数据加载器
        logdir = self.FLAGS.SOLVER.logdir  # 获取日志目录

        # check
        larger_than_191 = version.parse(torch.__version__) > version.parse('1.9.1')  # 检查PyTorch版本
        if not larger_than_191:  # 如果版本过低
            print('此功能仅适用于Pytorch>1.9.1。')  # 输出提示
            return  # 直接返回

        # profile
        batch = next(iter(self.train_loader))  # 获取一个训练批次
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)  # 配置profile调度
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA, ]
        with torch.profiler.profile(
                activities=activities, schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
                record_shapes=True, profile_memory=True, with_stack=True,
                with_modules=True) as prof:
            for _ in range(5):
                output = self.train_step(batch)
                output['train/loss'].backward()
                prof.step()

        print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=10)
                  .table(sort_by="cuda_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=10)
                  .table(sort_by="cuda_memory_usage", row_limit=10))

    def run(self):
        eval('self.%s()' % self.FLAGS.SOLVER.run)  # 动态调用指定的运行方法

    @classmethod
    def update_configs(cls):
        pass  # 用于更新配置参数（此处为空实现）

    @classmethod
    def worker(cls, rank, FLAGS):
        '''多进程/多卡训练的单进程入口，负责分布式初始化和主流程调用'''
        gpu = FLAGS.SOLVER.gpu  # 获取GPU列表
        torch.cuda.set_device(gpu[rank])  # 设置当前进程使用的GPU
        world_size = len(gpu)  # 获取总进程数（即GPU数量）
        if world_size > 1:  # 如果是多卡训练
            url = 'tcp://localhost:%d' % FLAGS.SOLVER.port  # 构造分布式通信地址
            torch.distributed.init_process_group(
                backend='nccl', init_method=url, world_size=world_size, rank=rank)  # 初始化分布式环境
        is_master = rank == 0  # 判断是否为主进程
        the_solver = cls(FLAGS, is_master)  # 实例化Solver对象
        the_solver.run()  # 调用run方法启动训练/测试流��

    @classmethod
    def main(cls):
        '''主入口，自动解析参数，支持单卡/多卡训练'''
        cls.update_configs()  # 更新配置参数
        FLAGS = parse_args()  # 解析命令行参数
        num_gpus = len(FLAGS.SOLVER.gpu)  # 获取GPU数量
        if num_gpus > 1:  # 多卡训练
            torch.multiprocessing.spawn(cls.worker, nprocs=num_gpus, args=(FLAGS,))  # 启动多进程，每个进程一个GPU
        else:
            cls.worker(0, FLAGS)  # 单卡训练，直接调用worker
