'''
@Author: Jiangtao
@Date: 2020-02-25 16:13:42
@LastEditors: Jiangtao
@LastEditTime: 2020-07-06 14:01:11
@Description: 
'''
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mixup_data(x, y, device,alpha=1.0,):

    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def randomMask(imgData,device):

    batchSize = imgData.shape[0]

    for i in range(batchSize):
        mask = np.random.uniform(low=0.0, high=1.0, size=(32,32))
        mask = torch.from_numpy(mask).to(device)
        x,y = np.random.randint(0, 4, size=(2,), dtype='l')
        imgData[i][0][(x)*32:(x+1)*32,(y)*32:(y+1)*32] = mask
    return imgData

def mixup_criterion(pred, y_a, y_b, lam):

    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)

if __name__ =='__main__':

    imgData1 = np.random.uniform(low=0.0, high=1.0, size=(1,1,128,128))
    print(imgData1[0][0])
    imgData2 = randomMask(imgData1)
    print(imgData2[0][0])


