'''
@Author: Jiangtao
@Date: 2019-12-10 18:55:59
@LastEditors: Jiangtao
@LastEditTime: 2020-07-07 16:21:45
@Description: 
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25,classes=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.classes = classes

    def forward(self,
                pred,
                target,
                weight=None,
                reduction='sum',
                avg_factor=None):
                            
        pred_sigmoid = pred.sigmoid()

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

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)


        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)

        # nn.BCEWithLogitsLoss()=nn.functional.binary_cross_entropy_with_logits()
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        loss1 = nn.BCELoss()


        return loss
        
class FocalLossWithSigmoid(nn.Module):
    
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLossWithSigmoid, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
 
    def __repr__(self):
        tmpstr = self._class.name_ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
 
    def forward(self, inputs, target):
        #inputs.size() B*C
        #target.size() B
        N = inputs.size(0) #batch size
        C = inputs.size(1) #class num

        dtype = target.dtype
        device = target.device
        t = target.unsqueeze(1)
        classRange = torch.arange(0, C, dtype=dtype, device=device).unsqueeze(0)
        posMask = (t == classRange)
        negMask = (t != classRange)
        sigmoidOut = torch.sigmoid(inputs)
        posSigmoidOut = sigmoidOut[posMask]  #只取正样本参与log计算，避免负样本干扰，此时负样本大多数概率为0，取log时为负无穷大，且log求导为1/p,p为0时，log导数也是无穷大
        negSigmoidOut = sigmoidOut[negMask]  #同上
        print('1',posMask.shape)
        print('2',negMask.shape)
        print('3',posSigmoidOut.shape)
        print('4',negSigmoidOut.shape)
        print('5',((1-posSigmoidOut)**self.gamma*torch.log(posSigmoidOut)).shape)
        posSigmoidOut = ((1-posSigmoidOut)**self.gamma*torch.log(posSigmoidOut)).sum()
        negSigmoidOut = (negSigmoidOut**self.gamma * torch.log(1 - negSigmoidOut)).sum()
        if np.any(np.isnan(posSigmoidOut.cpu().detach().numpy())):
            print("posSigmoidOut has nan")
            sys.exit()
        if np.any(np.isnan(negSigmoidOut.cpu().detach().numpy())):
            print("posSigmoidOut has nan")
            sys.exit()
        focalLossOut = -posSigmoidOut * self.alpha - negSigmoidOut * (1 - self.alpha)
        return focalLossOut

class FocalLossWithSigmoidMultiLabel(nn.Module):
    def __init__(self,classes,gamma=2, alpha=0.25):
        super(FocalLossWithSigmoidMultiLabel, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.classes = classes

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

    def forward(self, inputs, target):
        #inputs.size()  B*C
        #target.size()  B
        N = inputs.size(0)   #batch size
        C = inputs.size(1)   #class num

        dtype = target.dtype
        device = target.device

        sigmoidOut = torch.sigmoid(inputs)

        # turn onehot
        if(1):
            N = inputs.size(0) #batch size
            C = inputs.size(1) #class num
            target = target.view((N,self.classes))
            target_onehot = torch.FloatTensor(N, C)

            # In your for loop
            device = target.device
            target_onehot = target_onehot.to(device)

            target_onehot.zero_()
            target_onehot.scatter_(1, target, 1)

            target_onehot = target_onehot.type_as(inputs)
            target = target_onehot

        posMask = target.long()
        negMask = 1 - posMask

        posSigmoidOut = torch.where(posMask.byte(), sigmoidOut, torch.full_like(sigmoidOut, 1))
        negSigmoidOut = torch.where(negMask.byte(), sigmoidOut, torch.full_like(sigmoidOut, 0))

        posSigmoidOut = posSigmoidOut.view(-1)
        negSigmoidOut = negSigmoidOut.view(-1)

        posSigmoidOut = ((1-posSigmoidOut)**self.gamma*torch.log(posSigmoidOut)).sum()
        negSigmoidOut = (negSigmoidOut**self.gamma * torch.log(1 - negSigmoidOut)).sum()

        focalLossOut = -posSigmoidOut * self.alpha - negSigmoidOut * (1 - self.alpha)
        
        return focalLossOut
   
if __name__ =='__main__':

    def test_focal_loss():

        print('test')
        if(1):
            loss1 = FocalLoss(classes=1)
            loss2 = FocalLossWithSigmoid()

            input = Variable(torch.rand((5,4)), requires_grad=True)
            target = torch.from_numpy(np.array([2,3,0,0,3]))

            output = loss1(input, target)
            print(output)
            output = loss2(input, target)
            print(output)

        if(0):
            loss1 = FocalLoss(classes=2)
            loss2 = FocalLossWithSigmoidMultiLabel(classes=2)

            input = Variable(torch.rand((5,4)), requires_grad=True)
            target = torch.from_numpy(np.array([[0,2],[0,2],[0,2],[0,2],[0,2]]))
  
            output = loss1(input, target)
            print(output)
            output = loss2(input, target)
            print(output)

    test_focal_loss()

