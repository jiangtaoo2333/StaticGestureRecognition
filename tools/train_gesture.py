#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import numpy as np 
import random
import math
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from torch.cuda.amp import autocast,GradScaler

from src.dataprocess import DatasetDMSAge
# from src.network import *
import src.network as network
from src.loss._l2_loss import *
from src.utils.useful import *
import cv2
import timm

def get_args():

    parser = argparse.ArgumentParser("MultiTaskOnFace build by Jiangtao")

    # 可复现性
    parser.add_argument("--reproductive", type=bool, default=True)

    # 数据集
    parser.add_argument("--imgDirTrain", type=str,nargs='+',
                        default=[],
                        help="The list of images Dir")
    parser.add_argument("--imgDirValid", type=str,nargs='+',
                        default=[],
                        help="The list of images Dir")

    parser.add_argument("--batchSize", type=int, 
                        default=128, help="The number of images per batch")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--imgSize", type=int, default=256)
    parser.add_argument("--numEpoches", type=int, default=1000)

    # 模型
    parser.add_argument("--modelName", type=str, 
                        default='multi_out_8_angle_new', help="The name of model")
    # 损失
    parser.add_argument("--lossName", type=str, 
                        default='mse', help="The name of loss")
    # 优化器
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)

    # 验证集
    parser.add_argument("--validInterval", type=int, default=0, help="Number of epoches between testing phases")

    # 早停参数
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    # 预训练模型
    parser.add_argument("--pretrained_type", type=str, default="start_from_scratch", choices=["model", 'pretrained', 'start_from_scratch'])
    parser.add_argument("--pretrained_model", type=str, default="./models/multiScale_all_1024.pkl")

    # model和log路径
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--saved_path", type=str, default="./models/")

    # 设置分布式
    parser.add_argument('-n', '--nodes', default=1, type=int, help='节点数', metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='每个节点的gpu数目')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='当前节点的rank')
    parser.add_argument('-useDistributed', '--distributed', default=False, type=bool, help='是否使用分布式训练')

    parser.add_argument("--trainingBranch", type=str,nargs='+',
                        default=['eye','base'],
                        help="The list of trainingBranches")

    parser.add_argument("--savingEpoches", type=int,
                        default=10,
                        help="The list of trainingBranches")

    args = parser.parse_args()

    return args

def train(gpu,args):


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
    if args.reproductive:
        reproductive()

    # construct dataset for train and test,0 is boy 1 is girl
    TrainDataset = DatasetDMSAge(args.imgDirTrain, size=args.imgSize, isTrain='train')
    testDataset = DatasetDMSAge(args.imgDirValid, size=args.imgSize, isTrain='val')

    if True == args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            TrainDataset,
            num_replicas=args.world_size,
            rank=rank
        )

        trainDataLoader = torch.utils.data.DataLoader(TrainDataset, 
                                                    batch_size=args.batchSize, 
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
                                                    batch_size=args.batchSize, 
                                                    shuffle=False,\
                                                    num_workers=0,\
                                                    pin_memory=True,
                                                    sampler=valid_sampler)   # note that we're passing the collate function here

    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset=TrainDataset,
                                            batch_size=args.batchSize,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True)

        validDataLoader = torch.utils.data.DataLoader(dataset=testDataset,
                                                    batch_size=args.batchSize,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    # construct model
    if torch.cuda.is_available():

        if "model" == args.pretrained_type:

            oldmodel = torch.load(args.pretrained_model)


            if(type(oldmodel) == OrderedDict):
                model = eval('network.' + args.modelName)()
                model.load_state_dict(oldmodel)
                model.cuda(gpu)
            else:
                model = oldmodel
                model.cuda(gpu)
            
            print('load model:{}'.format(args.pretrained_model))

        elif "pretrained" == args.pretrained_type:
            model = eval('network.' + args.modelName)()
            oldmodel = torch.load(args.pretrained_model,map_location='cuda:0')
            
            network_dict = model.state_dict()
            oldmodel_dict = oldmodel.state_dict()

            new_state_dict = OrderedDict()	
            new_state_dict = {k:v for k,v in oldmodel_dict.items() if ('base' in k)}

            network_dict.update(new_state_dict)
            model.load_state_dict(network_dict)

            model.cuda(gpu)
            print('load pretrained model:{}'.format(args.pretrained_model))

        elif "start_from_scratch" == args.pretrained_type:
            model = eval('network.' + args.modelName)()
            model.cuda(gpu)
            print('start_from_scratch:{}'.format(args.modelName))

        else:
             sys.exit(123)

    model = timm.create_model(config.MODEL.NAME, pretrained=config.MODEL.PRETRAINED, num_classes=config.MODEL.NUM_CLASSES,
                                in_chans=config.MODEL.INPUT_CHANNEL)
    
    ###############################################################
    # 设置需要训练的参数
    setUpTrainingBrach(model,args)
    # model.train()
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
    criterion = cls_Loss(gpu)
    # criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = mse_Loss(gpu)

    #设置优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    # 设置学习率变化
    learning_rate_schedule = {"0": 1e-5, "1": 1e-4, "2": 1e-5,
                              "10": 5e-6, "25": 1e-6}

    #每个epoch迭代的次数
    num_iter_per_epoch = len(trainDataLoader)

    # 用于辅助年龄计算
    idx_tensor = [idx for idx in range(26)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
    softmax = nn.Softmax(dim=1).cuda(gpu)

    # 开始训练
    for epoch in range(args.numEpoches):
        
        #开始训练
        setUpTrainingBrach(model,args)
        #动态更新学习率
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]

        #从生成器中迭代生成数据
        for iter, batch in enumerate(trainDataLoader):
            
            imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
            label = batch[1].to(torch.device("cuda:{}".format(gpu)))
            constLabel = batch[2].to(torch.device("cuda:{}".format(gpu)))

            predict = model(imgBatch)

            # 分类损失
            loss_cls = criterion(predict[12],label)

            # 回归损失
            age_predict = softmax(predict[12])
            # print('age_predict.shape:',age_predict.shape)
            age_predict = torch.sum(age_predict * idx_tensor, 1) * 3
            loss_reg = reg_criterion(age_predict, constLabel)
            # print('age_predict.shape:',age_predict.shape)
            # print('constLabel.shape:',constLabel.shape)
            # time.sleep(1000)

            # 损失合并
            loss = loss_cls + 2 * loss_reg
            # loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # cal accuracy
            correctList = []
            _, preds = predict[12].max(1)
            correct = preds.eq(label.reshape((-1)).long()).sum()
            correct = correct.float() / imgBatch.shape[0]
            correctList.append(correct.item())
            correctArray = np.array(correctList)

            if 0 == rank:
                writer.add_scalar('loss_iter', loss.data, global_step=num_iter_per_epoch*epoch + iter)
                writer.add_scalar('accuracy', correctArray.mean(), global_step=num_iter_per_epoch*epoch + iter)
                if iter % 5 == 0:
                    print("Epoch: {}/{} ,iteration: {}/{} ,loss: {} ,accuracy:{}".format(epoch+1, args.numEpoches, iter+1, num_iter_per_epoch,loss.data,correctArray.mean()))

        if 0 == rank:
            writer.add_scalar('EpochEnd', loss.data, global_step=epoch)
            if(epoch % args.savingEpoches == 0):
                if True == args.distributed:
                    try:
                        torch.save(model.module.state_dict(), args.saved_path + os.sep + "{}_DMS_age_{}.pkl".format(type(model).__name__,epoch),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.module.state_dict(), args.saved_path + os.sep + "{}_DMS_age_{}.pkl".format(type(model).__name__,epoch))
                else:
                    try:
                        torch.save(model.state_dict(), args.saved_path + os.sep + "{}_DMS_age_{}.pkl".format(type(model).__name__,epoch),_use_new_zipfile_serialization=False)
                    except:
                        torch.save(model.state_dict(), args.saved_path + os.sep + "{}_DMS_age_{}.pkl".format(type(model).__name__,epoch))

        # 验证集
        # 对齐验证集容易爆内存，所以使用下面方法关闭验证集，非对齐可开启验证集
        validInterval = args.validInterval if args.validInterval > 0 else sys.maxsize
        if (0 == rank) and ((epoch+1) % validInterval == 0):

            model.eval()
            lossAll = []
            correctList = []

            for iter, batch in enumerate(validDataLoader):

                imgBatch = batch[0].to(torch.device("cuda:{}".format(gpu))).float()
                label = batch[1].to(torch.device("cuda:{}".format(gpu)))
                constLabel = batch[2].to(torch.device("cuda:{}".format(gpu)))

                predict = model(imgBatch)

                # 分类损失
                loss_cls = criterion(predict[12],label)

                # 回归损失
                age_predict = softmax(predict[12])
                age_predict = torch.sum(age_predict * idx_tensor, 1) * 3
                loss_reg = reg_criterion(age_predict, constLabel)

                # 损失合并
                loss = loss_cls + 2 * loss_reg

                # 只看回归损失
                lossAll.append(loss_reg.cpu().detach().numpy())

                # cal accuracy
                _, preds = predict[12].max(1)
                correct = preds.eq(label.reshape((-1)).long()).sum()
                correct = correct.float() / imgBatch.shape[0]
                correctList.append(correct.item())

            lossAll = np.array(lossAll)
            loss = lossAll.mean(axis=0)
            correctArray = np.array(correctList)

            print("Epoch: {}/{} ,valid loss: {} ,accuracy:{}".format(epoch+1, args.numEpoches, loss, correctArray.mean()))


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

