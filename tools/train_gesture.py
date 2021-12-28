# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-10-08 10:31:28
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-12-27 16:13:56
* @Description  : 性别训练接口
'''
#!/usr/bin/env python

import argparse
import math
import os
import random
import shutil
import sys
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
from collections import OrderedDict

import cv2
import mmcv
# from src.network import *
import src.network as network
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from mmcv import Config
from src.dataprocess import (AugMixDataset, DatasetGesture,DatasetGestureSim)
from src.dataprocess.transform import cutmix_data, mixup_criterion, mixup_data
from src.loss._l2_loss import *
from src.scheduler import GradualWarmupScheduler
from src.utils.useful import *
from tensorboardX import SummaryWriter
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from thop import profile,clever_format

def get_args():

    parser = argparse.ArgumentParser("MultiTaskOnFace build by Jiangtao")

    parser.add_argument('config', help='train config file path')

    # 设置分布式
    parser.add_argument('-n', '--nodes', default=1, type=int, help='节点数', metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='每个节点的gpu数目')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='当前节点的rank')
    parser.add_argument('-useDistributed', '--distributed', default=False, type=bool, help='是否使用分布式训练')


    args = parser.parse_args()

    return args

def train(gpu,args):

    cfg = Config.fromfile(args.config)
    if cfg.saved_path == None:
        cfg.saved_path = './work_dirs/{}'.format(osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.saved_path))

    # 分布式
    if True == args.distributed:
        rank = args.nr * args.gpus + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='tcp://localhost:23456',                                   
            world_size=args.world_size,                              
            rank=rank                                               
        )

    else:
        rank = 0

    torch.manual_seed(0)                                                    
    torch.cuda.set_device(gpu)

    # tensorboard
    if 0 == rank:
        timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dir = './logs/{}_{}'.format(os.path.basename(__file__).split('.')[0],timeNow)
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except:
                pass
        print('logg is in {}'.format(dir))
        writer = SummaryWriter(dir)

    # 确定可复现性
    if cfg.reproductive:
        reproductive()

    # construct dataset for train and test,0 is boy 1 is girl
    TrainDataset = DatasetGesture(cfg.imgDirTrain, cfg.imgTxtTrain, size=cfg.imgSize, isTrain='train')
    testDataset = DatasetGesture(cfg.imgDirValid, cfg.imgTxtValid, size=cfg.imgSize, isTrain='val')

    if cfg.get('augmix_alpha',0) > 0:
        TrainDataset = DatasetGestureSim(cfg.imgDirTrain, cfg.imgTxtTrain, size=cfg.imgSize, isTrain='train')
        TrainDataset = AugMixDataset(TrainDataset)

    if True == args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            TrainDataset,
            num_replicas=args.world_size,
            rank=rank
        )

        trainDataLoader = torch.utils.data.DataLoader(TrainDataset, 
                                                    batch_size=cfg.batchSize, 
                                                    shuffle=False,\
                                                    num_workers=0,\
                                                    pin_memory=True,
                                                    sampler=train_sampler)  # note that we're passing the collate function here

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            testDataset,
            num_replicas=args.world_size,
            rank=rank
        )

        validDataLoader = torch.utils.data.DataLoader(testDataset, 
                                                    batch_size=cfg.batchSize, 
                                                    shuffle=False,\
                                                    num_workers=0,\
                                                    pin_memory=True,
                                                    sampler=valid_sampler)   # note that we're passing the collate function here

    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset=TrainDataset,
                                            batch_size=cfg.batchSize,
                                            shuffle=True,
                                            num_workers=cfg.workers,
                                            pin_memory=True)

        validDataLoader = torch.utils.data.DataLoader(dataset=testDataset,
                                                    batch_size=cfg.batchSize,
                                                    shuffle=True,
                                                    num_workers=cfg.workers,
                                                    pin_memory=True)

    # construct model
    model = timm.create_model(cfg.modelName, pretrained=cfg.pretrained, num_classes=cfg.numClasses,
                                in_chans=cfg.channels).cuda()
    input = torch.randn(1, 1, 128, 128).cuda()
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {} params: {}'.format(flops,params))
    ###############################################################
    # 设置需要训练的参数
    # setUpTrainingBrach(model,cfg)
    # setUpbias(model,cfg)
    model.train()
    ###############################################################


    ###############################################################
    # Wrap the model
    if True == args.distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[gpu])
    else:
        pass
    ###############################################################

    #定制loss函数
    eps = cfg.get('labelSmooth', 0)
    criterion = cls_Loss(gpu,cfg.lossName,eps)

    #设置优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    # 设置学习率变化,warmupStep或者cosine annealing
    scheduler_steplr = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

    scheduler_cosine = CosineAnnealingLR(optimizer,T_max=20)

    #每个epoch迭代的次数
    num_iter_per_epoch = len(trainDataLoader)

    correctBest = 0
    lossBest = 9999
    epochBest = 0
    # 开始训练
    for epoch in range(1,cfg.numEpoches+1):

        #开始训练
        model.train()

        #动态更新学习率
        schedulerUsed = cfg.get('scheduler', 'stepLrWithWarmUp')
        if 'stepLrWithWarmUp' == schedulerUsed:
            scheduler_warmup.step(epoch)
        if 'cosineAnnealing' == schedulerUsed:
            scheduler_cosine.step()

        #从生成器中迭代生成数据
        for iter, batch in enumerate(trainDataLoader):

            if cfg.get('mixup_alpha',0) > 0 and random.random() > 0.5:
                imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
                groundtruth = batch[1].to(torch.device("cuda:{}".format(gpu)))
                inputs, targets_a, targets_b, lam = mixup_data(imgBatch, groundtruth,
                                                       cfg.get('mixup_alpha',0), True)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
                predict = model(inputs)
                loss = mixup_criterion(criterion,predict,targets_a,targets_b,lam)
            elif cfg.get('cutmix_alpha',0) > 0 and random.random() > 0.5:
                imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
                groundtruth = batch[1].to(torch.device("cuda:{}".format(gpu)))
                inputs, targets_a, targets_b, lam = cutmix_data(imgBatch, groundtruth,
                                                       cfg.get('cutmix_alpha',0), True)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
                predict = model(inputs)
                loss = mixup_criterion(criterion,predict,targets_a,targets_b,lam)
            elif cfg.get('augmix_alpha',0) > 0:
                images = batch[0] #[(batchsize,1,128,128)] * 3
                imgBatch = torch.cat(images, 0).cuda().float() #[(batchsize*3,1,128,128)]
                groundtruth = batch[1].to(torch.device("cuda:{}".format(gpu)))
                predict = model(imgBatch)
                logits_all = predict
                logits_clean, logits_aug1, logits_aug2 = torch.split(
                            logits_all, images[0].size(0))

                loss = criterion(logits_clean,groundtruth)

                p_clean, p_aug1, p_aug2 = F.softmax(
                            logits_clean, dim=1), F.softmax(
                                logits_aug1, dim=1), F.softmax(
                                    logits_aug2, dim=1)
                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            else:
                imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
                groundtruth = batch[1].to(torch.device("cuda:{}".format(gpu)))
                predict = model(imgBatch)
                loss = criterion(predict,groundtruth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # cal accuracy
            correctList = []
            _, preds = predict[0:groundtruth.shape[0]].max(1)
            correct = preds.eq(groundtruth.reshape((-1)).long()).sum()
            correct = correct.float() / groundtruth.shape[0]
            correctList.append(correct.item())
            correctArray = np.array(correctList)

            if 0 == rank:
                writer.add_scalar('loss_iter', loss.data, global_step=num_iter_per_epoch*epoch + iter)
                writer.add_scalar('accuracy', correctArray.mean(), global_step=num_iter_per_epoch*epoch + iter)
                if iter % 5 == 0:
                    print("Epoch: {}/{} ,iteration: {}/{} ,lr: {} ,loss: {:.5f} ,accuracy:{:.5f}".format(epoch, cfg.numEpoches, iter+1, num_iter_per_epoch, optimizer.param_groups[0]['lr'], loss.data,correctArray.mean()))

        if 0 == rank:
            writer.add_scalar('EpochEnd', loss.data, global_step=epoch)
            if(epoch % cfg.savingEpoches == 0):
                if True == args.distributed:
                    try:
                        torch.save(model.module.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender_{}.pkl".format(type(model).__name__,epoch),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.module.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender{}.pkl".format(type(model).__name__,epoch))
                else:
                    try:
                        torch.save(model.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender{}.pkl".format(type(model).__name__,epoch),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender{}.pkl".format(type(model).__name__,epoch))

        # 验证集
        # 对齐验证集容易爆内存，所以使用下面方法关闭验证集，非对齐可开启验证集
        if (0 == rank) and (epoch % (cfg.validInterval if cfg.validInterval > 0 else sys.maxsize) == 0):
            
            model.eval()
            lossAll = []
            correctList = []

            for iter, batch in enumerate(validDataLoader):

                imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
                groundtruth = batch[1].to(torch.device("cuda:{}".format(gpu)))

                predict = model(imgBatch)

                loss = criterion(predict,groundtruth)

                lossAll.append(loss.cpu().detach().numpy())

                # cal accuracy
                _, preds = predict.max(1)
                correct = preds.eq(groundtruth.reshape((-1)).long()).sum()
                correct = correct.float() / imgBatch.shape[0]
                correctList.append(correct.item())
            

            lossAll = np.array(lossAll)
            loss = lossAll.mean(axis=0)
            correctArray = np.array(correctList)
            correctmean = correctArray.mean()
            if(correctmean > correctBest):
                correctBest = correctmean
                epochBest = epoch
                if True == args.distributed:
                    try:
                        torch.save(model.module.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender_best_{}.pkl".format(type(model).__name__,correctBest),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.module.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender_best_{}.pkl".format(type(model).__name__,correctBest))
                else:
                    try:
                        torch.save(model.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender_best_{}.pkl.pkl".format(type(model).__name__,correctBest),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.state_dict(), cfg.saved_path + os.sep + "{}_DMS_gender_best_{}.pkl".format(type(model).__name__,correctBest))
            
            print("Epoch: {}/{} ,valid loss: {:.5f} ,accuracy:{:.5f}".format(epoch, cfg.numEpoches, loss, correctArray.mean()))

    print('best epoch: {} best accuracy: {:.5f}'.format(epochBest,correctBest))

def main():

    args = get_args()
    print('-----------------start---------------')
    print(args)
    print('-------------------------------------')
    args.gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if(1 == args.world_size):
        args.distributed = False
    else:
        args.distributed = False

    print('distributed:',args.distributed)

    if True == args.distributed:
        os.environ['MASTER_ADDR'] = '10.57.23.164'  
        port = findPort(15646)    
        os.environ['MASTER_PORT'] = '{}'.format(port)

        mp.spawn(train, nprocs=args.gpus, args=(args,))

    else:
        train(0,args)


if __name__ == "__main__":

    main()

