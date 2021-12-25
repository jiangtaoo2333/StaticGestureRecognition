'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-12-24 10:58:31
@Description: 
'''
import os
import sys
import time


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .focal_loss import FocalLoss, FocalLossWithSigmoid
from .ghm_loss import GHMC
from .wing_loss import WingLoss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .label_smothing_cross_entropy_loss import LabelSmoothCrossEntropyLoss

class cls_Loss_HardMining(nn.Module):

    def __init__(self,pred_for,target_for):
        super(cls_Loss_HardMining, self).__init__()
        self.pred_for = pred_for
        self.target_for = target_for
    def forward(self,pred,target):
        '''
        pred n*c    (network output)
        target n*1   (need onehot label)
        pred_index and target_index should not be equal
        '''

        # turn onehot
        if(1):
            N = pred.size(0) #batch size
            C = pred.size(1) #class num
            target = target.view((N,1))
            target_onehot = torch.FloatTensor(N, C)

            # In your for loop
            device = target.device
            target_onehot = target_onehot.to(device)

            target_onehot.zero_()
            target_onehot.scatter_(1, target.long(), 1)

            target_onehot = target_onehot.type_as(pred)
            target = target_onehot

        # find pred max and target max:N,N
        pred_index = torch.max(pred,1)[1]
        target_index = torch.max(target,1)[1]
        
        # base scale:N,C
        scale_0 = torch.ones_like(pred)
        
        # find if pred_max==pred_for,if so,put 1,else 0
        pred_0 = pred_index == self.pred_for
        tartget_1 = target_index == self.target_for

        # add pred_0 and target_1.some will equal 2,if so,put 1,else 0.
        scale_1 = (pred_0 + tartget_1).type_as(scale_0)
        scale_1 = torch.gt(scale_1,torch.ones_like(scale_1))

        # resize to N,C
        scale_1 = scale_1.reshape((-1,1)).expand_as(pred).type_as(scale_0)

        # scale = scale0 + 1:N,C
        scale = scale_0+scale_1

        # loss:N,C scale:N,C
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction = 'none')
        loss = loss * scale
        loss = loss.sum()

        return loss
        
class mse_Loss(nn.Module):
    
    def __init__(self,deviceId):
        super(mse_Loss, self).__init__()

        self.mse = nn.MSELoss(reduction='none')
        self.mse = self.mse.to(torch.device("cuda:{}".format(deviceId)))

    def forward(self, output, target):

        loss = 0
        # 眼睛
        target = target.view_as(output).float()
        loss += self.mse(output, target).sum()

        # 损失均值化
        loss = loss / float(output.size()[0])
    
        return loss

class cls_Loss(nn.Module):
    
    def __init__(self,deviceId,lossname='cross entroy',eps=0):
        super(cls_Loss, self).__init__()

        assert lossname in ['focalLoss','crossEntroy','ghmLoss']
        self.lossname = lossname
        self.eps = eps

        self.gpu_id = deviceId
        self.CrossEntropyLoss = nn.CrossEntropyLoss().to(torch.device("cuda:{}".format(deviceId)))
        self.FocalLoss = FocalLoss()
        self.GHMloss = GHMC()
        self.CrossEntropyLossWithlableSmooth = LabelSmoothCrossEntropyLoss(smoothing=self.eps)

    def forward(self, output, target):
        '''
        output n,c
        target n,1
        '''
        # print(type(target))
        target = target.float()
        target = Variable(target,requires_grad=True).cuda(self.gpu_id).long()
        loss = 0
        if(self.eps == 0):
            if(self.lossname == 'focalLoss'):
                loss += self.FocalLoss(output, target)
            if(self.lossname == 'crossEntroy'):
                loss += self.CrossEntropyLoss(output, target.squeeze())
            if(self.lossname == 'ghmLoss'):
                loss += self.GHMloss(output, target)
        if(self.eps > 0):
            if(self.lossname == 'crossEntroy'):
                loss += self.CrossEntropyLossWithlableSmooth(output, target.squeeze())

        # 损失均值化
        # loss = loss / float(output.size()[0])

        return loss

class mse_Loss_optimize_mask(nn.Module):
    
    def __init__(self,deviceId):
        super(mse_Loss_optimize_mask, self).__init__()

        self.mse = nn.MSELoss(reduction='none')
        self.mse = self.mse.to(torch.device("cuda:{}".format(deviceId)))

    def forward(self, output, target):

        loss = 0
        # 眼睛
        target = target.view_as(output).float()
        loss += self.mse(output, target)
        a= loss > loss.mean()
        loss = loss[a]

        # 损失均值化
        loss = loss.sum() / float(output.size()[0])
    
        return loss

class mse_Loss_optimize_angle(nn.Module):
    
    def __init__(self,deviceId):
        super(mse_Loss_optimize_angle, self).__init__()

        self.mse = nn.MSELoss(reduction='none')
        self.mse = self.mse.to(torch.device("cuda:{}".format(deviceId)))

    def forward(self, output, target):

        loss = 0

        # (B,22),calculate loss
        target = target.view_as(output).float()
        loss += self.mse(output, target)

        # (B),calculate 2 scale and merge them to one
        if(22 == target.shape[1]):
            left = target[:,0]
            mid = target[:,10]
            right = target[:,20]
            
        if(26 == target.shape[1]):
            left = target[:,0]
            mid = target[:,12]
            right = target[:,24]


        
        abs = torch.abs
        scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
        scale2 = abs((right-mid) / ((mid-left)+1e-5))

        scale1 = torch.where(scale1 < 0, torch.full_like(scale1, 1), scale1)
        scale2 = torch.where(scale2 < 0, torch.full_like(scale2, 1), scale2)
        scale1 = torch.where(scale1 > 100, torch.full_like(scale1, 100), scale1)
        scale2 = torch.where(scale2 > 100, torch.full_like(scale2, 100), scale2)

        scale = torch.where(scale1>scale2,scale1,scale2)
        scale = torch.sqrt(scale+1e-5)
        # scale = torch.log(scale)
        
        loss = torch.sum(loss,dim=1)
        loss *= scale

        # 损失均值化
        loss = loss.sum() / float(output.size()[0])
    
        return loss

class mse_Loss_optimize_angle_mouth(nn.Module):
    
    def __init__(self,deviceId):
        super(mse_Loss_optimize_angle_mouth, self).__init__()

        self.mse = nn.MSELoss(reduction='none')
        self.mse = self.mse.to(torch.device("cuda:{}".format(deviceId)))

    def forward(self, output, target, eyeGt):

        loss = 0

        # (B,22),calculate loss
        target = target.view_as(output).float()
        loss += self.mse(output, target)

        # (B),calculate 2 scale and merge them to one
        eyeGt = eyeGt.view((target.shape[0],-1)).float()
        left = eyeGt[:,0]
        mid = eyeGt[:,10]
        right = eyeGt[:,20]
        
        abs = torch.abs
        scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
        scale2 = abs((right-mid) / ((mid-left)+1e-5))

        scale1 = torch.where(scale1 < 0, torch.full_like(scale1, 1), scale1)
        scale2 = torch.where(scale2 < 0, torch.full_like(scale2, 1), scale2)
        scale1 = torch.where(scale1 > 100, torch.full_like(scale1, 100), scale1)
        scale2 = torch.where(scale2 > 100, torch.full_like(scale2, 100), scale2)

        scale = torch.where(scale1>scale2,scale1,scale2)
        scale = torch.sqrt(scale+1e-5)
        # scale = torch.log(scale)
        
        loss = torch.sum(loss,dim=1)
        loss *= scale

        # 损失均值化
        loss = loss.sum() / float(output.size()[0])
    
        return loss