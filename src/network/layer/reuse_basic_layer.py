# -*- coding: UTF-8 -*-
'''
 * @Author: Jiangtao
 * @Email: jiangtaoo2333@163.com
 * @Company: Streamax
 * @Date: 2019/08/05 14:03
 * @Description: 
'''
import logging as log
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from ._basic_layer import *
from .am_softmax import AMSoftmax
from .arc_softmax import ArcSoftmax
from .circle_softmax import CircleSoftmax

class backBone(nn.Module):
    def __init__(self):
        super(backBone, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class backBone0104(nn.Module):
    def __init__(self):
        super(backBone0104, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #16
        layer5 = Conv2dBatchReLU(128, 128, 3, 1)
        layer6 = Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone(nn.Module):
    def __init__(self):
        super(eyeBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 22)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone_Onet(nn.Module):
    def __init__(self):
        super(eyeBone_Onet, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 1)  #64
        layer1= nn.MaxPool2d(kernel_size=(3,3),stride=2,ceil_mode=True)
        layer2 = Conv2dBatchReLU(32, 64, 3, 1)  #32
        layer3= nn.MaxPool2d(kernel_size=(3,3),stride=2,ceil_mode=True)
        layer4 = Conv2dBatchReLU(64, 128, 3, 1)  #16
        layer5= nn.MaxPool2d(kernel_size=(3,3),stride=2,ceil_mode=True)
        layer6 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        layer7 = GlobalAvgPool2d()
        layer8 = FullyConnectLayer(256, 22)
        

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7,
            layer8
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone_OnetV2(nn.Module):
    def __init__(self):
        super(eyeBone_OnetV2, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #32
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #16
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #8

        layer5 = Conv2dBatchReLU(128, 128, 3, 1)  #8
        layer6 = Conv2dBatchReLU(128, 256, 3, 2)  #4
        layer7 = GlobalAvgPool2d()
        layer8 = FullyConnectLayer(256, 22)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7,
            layer8
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class baseBone_Onet(nn.Module):
    def __init__(self):
        super(baseBone_Onet, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #32
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #16
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #8

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class alignBone_Onet(nn.Module):
    def __init__(self):
        super(alignBone_Onet, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #8
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #4
        layer2 = GlobalAvgPool2d()
        layer3 = FullyConnectLayer(256, 22)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class binaryBone_Onet(nn.Module):
    def __init__(self):
        super(binaryBone_Onet, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #8
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #4
        layer2 = GlobalAvgPool2d()
        layer3 = FullyConnectLayer(256, 2)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class gazeBone_Onet(nn.Module):
    def __init__(self):
        super(gazeBone_Onet, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #8
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #4
        layer2 = GlobalAvgPool2d()
        layer3 = FullyConnectLayer(256, 3)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone_multiscale(nn.Module):

    def __init__(self):
        super(eyeBone_multiscale, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 22)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class eyeBone_multiscale_20210406(nn.Module):

    def __init__(self):
        super(eyeBone_multiscale_20210406, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 18)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class eyeBone_multiscale_1111(nn.Module):

    def __init__(self):
        super(eyeBone_multiscale_1111, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 26)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class eyeBone_elementwise(nn.Module):
    
    def __init__(self):

        super(eyeBone_elementwise, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 128, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(128, 128, 3, 2)  #4
        self.layer3 = nn.Conv2d(128, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(128, 22)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        elementAdd = out1 + out2 + out3

        landmarks = self.fc(elementAdd)

        return landmarks

class mouthBone(nn.Module):
    def __init__(self):
        super(mouthBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 16)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class mouthBone_multiscale(nn.Module):
    def __init__(self):
        super(mouthBone_multiscale, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 16)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class mouthBone_elementwise(nn.Module):
    
    def __init__(self):

        super(mouthBone_elementwise, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 128, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(128, 128, 3, 2)  #4
        self.layer3 = nn.Conv2d(128, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(128, 16)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        elementAdd = out1 + out2 + out3
        
        landmarks = self.fc(elementAdd)

        return landmarks
        
class faceBone(nn.Module):
    def __init__(self):
        super(faceBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 4)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x      

class binaryFaceBone(nn.Module):
    def __init__(self):
        super(binaryFaceBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 2)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class alignQualityBone(nn.Module):
    def __init__(self):
        super(alignQualityBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 1)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class detectBone(nn.Module):
    def __init__(self):
        super(detectBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 4)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class detectBone_multiScale(nn.Module):

    def __init__(self):
        super(detectBone_multiScale, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 4)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        detectRet = self.fc(multi_scale)

        return detectRet

class emotionBone(nn.Module):

    def __init__(self):
        super(emotionBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)


        layer2 = Conv2dBatchReLU(256, 256, 3, 1)
        layer3 = Conv2dBatchReLU(256, 512, 3, 2)


        layer4 = Conv2dBatchReLU(512, 512, 3, 1)
        layer5 = Conv2dBatchReLU(512, 512, 3, 2)


        layer6 = GlobalAvgPool2d()
        layer7 = FullyConnectLayer(512, 2)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class emotionBone_WITH_BG(nn.Module):

    def __init__(self):
        super(emotionBone_WITH_BG, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)


        layer2 = Conv2dBatchReLU(256, 256, 3, 1)
        layer3 = Conv2dBatchReLU(256, 512, 3, 2)


        layer4 = Conv2dBatchReLU(512, 512, 3, 1)
        layer5 = Conv2dBatchReLU(512, 512, 3, 2)


        layer6 = GlobalAvgPool2d()
        layer7 = FullyConnectLayer(512, 3)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class emotionBone_ArcFace(nn.Module):
    
    def __init__(self):
        super(emotionBone_ArcFace, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)


        layer2 = Conv2dBatchReLU(256, 256, 3, 1)
        layer3 = Conv2dBatchReLU(256, 512, 3, 2)


        layer4 = Conv2dBatchReLU(512, 512, 3, 1)
        layer5 = Conv2dBatchReLU(512, 512, 3, 2)

        layer6 = GlobalAvgPool2d()

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class glassBone(nn.Module):

    def __init__(self):
        super(glassBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        # layer0 = Conv2dBatchReLU_BlurPool(128, 128, 3, 1)
        # layer0 = Conv2dBatchReLU_MaxPool(128, 128, 3, 1)

        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        # layer1 = Conv2dBatchReLU_BlurPool(128, 256, 3, 2)  #8
        # layer1 = Conv2dBatchReLU_MaxPool(128, 256, 3, 2)

        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        # layer2 = Conv2dBatchReLU_BlurPool(256, 256, 3, 2)  #4
        # layer2 = Conv2dBatchReLU_MaxPool(5122, 256, 3, 2)

        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 5)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class glassBone_centerloss(nn.Module):
    def __init__(self):
        super(glassBone_centerloss, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)

        layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8

        layer2 = Conv2dBatchReLU(256, 256, 3, 2)  #4

        layer3 = GlobalAvgPool2d()

        layer4 = FullyConnectLayer(256, 2)

        self.layer5 = FullyConnectLayer(2, 4)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)
        y = self.layer5(x)

        return x,y

class glassBone_deeper(nn.Module):
    def __init__(self):
        super(glassBone_deeper, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 1)


        layer2 = Conv2dBatchReLU(256, 256, 3, 1)
        layer3 = Conv2dBatchReLU(256, 512, 3, 2)


        layer4 = Conv2dBatchReLU(512, 512, 3, 1)
        layer5 = Conv2dBatchReLU(512, 512, 3, 2)


        layer6 = GlobalAvgPool2d()
        layer7 = FullyConnectLayer(512, 5)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class glassBone_more(nn.Module):

    def __init__(self):
        super(glassBone_more, self).__init__()

        layer0 = Conv2dBatchReLUConv2d(128, 128, 3, 1)

        layer1 = Conv2dBatchReLUConv2d(128, 256, 3, 2)

        layer2 = Conv2dBatchReLUConv2d(256, 256, 3, 2)


        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(256, 5)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class faceBone_blurpool(nn.Module):

    def __init__(self):
        super(faceBone_blurpool, self).__init__()

        layer0 = Conv2dBatchReLU_BlurPool(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU_BlurPool(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU_BlurPool(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 4)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x 

class eyeHeatmap(nn.Module):

    def __init__(self):

        super(eyeHeatmap, self).__init__()

        layer0 = Conv2dBatchReLU(128, 32, 3, 1)
        layer1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        layer2 = Conv2dBatchReLU(32, 11, 3, 1)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class mouthHeatmap(nn.Module):

    def __init__(self):

        super(mouthHeatmap, self).__init__()

        layer0 = Conv2dBatchReLU(128, 32, 3, 1)
        layer1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        layer2 = Conv2dBatchReLU(32, 8, 3, 1)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class eyeBone_multiscale_heatmap(nn.Module):

    def __init__(self):
        super(eyeBone_multiscale_heatmap, self).__init__()

        self.layer0 = Conv2dBatchReLU(128+11, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 22)

    def forward(self, x, heatmap):

        heatmap = nn.functional.interpolate(heatmap, scale_factor=0.5, mode='bilinear', align_corners=False) 
        x = torch.cat([x, heatmap], 1)

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class mouthBone_multiscale_heatmap(nn.Module):

    def __init__(self):
        super(mouthBone_multiscale_heatmap, self).__init__()

        self.layer0 = Conv2dBatchReLU(128+8, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 16)

    def forward(self, x, heatmap):

        heatmap = nn.functional.interpolate(heatmap, scale_factor=0.5, mode='bilinear', align_corners=False) 
        x = torch.cat([x, heatmap], 1)

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

class detectBone_SSD(nn.Module):

    def __init__(self):

        super(detectBone_SSD, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 128, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(128, 128, 3, 2)  #4
        self.layer3 = nn.Conv2d(128, 128, 4, 1, 0)     #1

        self.layer0_reg = nn.Conv2d(128, 4, 3, 1, 1)
        self.layer1_reg = nn.Conv2d(128, 4, 3, 1, 1)
        self.layer2_reg = nn.Conv2d(128, 4, 3, 1, 1)
        self.layer3_reg = nn.Conv2d(128, 4, 3, 1, 1)

        self.layer0_cls = nn.Conv2d(128, 2, 3, 1, 1)
        self.layer1_cls = nn.Conv2d(128, 2, 3, 1, 1)
        self.layer2_cls = nn.Conv2d(128, 2, 3, 1, 1)
        self.layer3_cls = nn.Conv2d(128, 2, 3, 1, 1)

    def forward(self, x):
        
        loc = []
        conf = []

        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        out0_reg = self.layer0_reg(out0)
        out1_reg = self.layer1_reg(out1)
        out2_reg = self.layer2_reg(out2)
        out3_reg = self.layer3_reg(out3)
        regList = [out0_reg,out1_reg,out2_reg,out3_reg]

        out0_cls = self.layer0_cls(out0)
        out1_cls = self.layer1_cls(out1)
        out2_cls = self.layer2_cls(out2)
        out3_cls = self.layer3_cls(out3)
        clsList = [out0_cls,out1_cls,out2_cls,out3_cls]

        for _reg,_cls in zip(regList, clsList):
            loc.append(_reg.permute(0, 2, 3, 1).contiguous())  
            conf.append(_cls.permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1 ,4)
        conf = conf.view(conf.size(0), -1, 2)

        return loc,conf

class detectBone_YOLO(nn.Module):

    def __init__(self):

        super(detectBone_YOLO, self).__init__()

        self.na = 3
        self.nc = 1
        self.isTraining = True
        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 128, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(128, 128, 3, 2)  #4

        self.layer_out = Conv2dBatchReLU(128, 6*self.na, 2, 1,False)  #7 #3
        self.create_everything()
        
    def create_everything(self,):

        anchors_List = [[20,20,40,40,60,60],[80,80,100,100,120,120]]
        grid_list = [(7,7),(3,3)]

        self.stride_List = [18.3,42.6]
        self.anchors = []
        self.anchor_vec = []
        self.anchor_wh = []
        self.grids = []
        
        for i in range(2):
            
            anchors = torch.Tensor(anchors_List[i]).reshape((3,2))
            stride = self.stride_List[i]

            self.anchors.append(anchors)
            self.anchor_vec.append(anchors/stride)
            self.anchor_wh.append((anchors/stride).view(1, self.na, 1, 1, 2))
            self.grids.append(self.create_grids(grid_list[i]))


    def create_grids(self, ng=(13, 13)):

        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)
        yv, xv = torch.meshgrid([torch.arange(self.ny), torch.arange(self.nx)])
        return torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

    def forward(self, x):

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)

        x_out_1 = self.layer_out(x1)  # 7*7
        x_out_2 = self.layer_out(x2)  # 3*3

        if self.isTraining:
            res = []
            for i,p in enumerate([x_out_1,x_out_2]):
                bs, _, ny, nx = p.shape
                p = p.view(bs, self.na, 6, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                res.append(p)

        else:
            res = []
            for i,p in enumerate([x_out_1,x_out_2]):
                
                bs, _, ny, nx = p.shape
                p = p.view(bs, self.na, 6, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                
                io = p.clone()  # inference output
                # print(io[..., :2].shape)
                # print(self.grids[i].shape)
                # print(self.anchor_wh[i].shape)
                # print(self.grids[i])
                # print(self.anchor_wh[i])
                # print(self.stride_List[i])
                # time.sleep(1000)
                io[..., :2] = torch.sigmoid(io[..., :2]) + self.grids[i].to(io.device)  # xy
                io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh[i].to(io.device)  # wh yolo method
                io[..., :4] *= self.stride_List[i]

                res.append(io.view(bs, -1, 6))

        return res

class PriorBox(object):
    """
    1、计算先验框，根据feature map的每个像素生成box;
    2、框的中个数为： 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
    3、 cfg: SSD的参数配置，字典类型
    """
    def __init__(self):
        super(PriorBox, self).__init__()
        self.img_size = 128
        self.feature_maps = [16,8,4,1]
        self.sizes = [30,50,80,100]
        self.steps = [8,16,32,128]
        self.variance = [0.1,0.2]

    def forward(self):
        mean = [] #用来存放 box的参数

        for k, f in enumerate(self.feature_maps):

            for i, j in product(range(f), repeat=2):


                f_k = self.img_size/self.steps[k]
                
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k

                s_k = self.sizes[k]/self.img_size

                mean += [cx, cy, s_k, s_k]

        boxes = torch.tensor(mean).view(-1, 4)

        boxes.clamp_(max=1, min=0)

        return boxes

class YawBone(nn.Module):

    def __init__(self):
        super(YawBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 66)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class PitchBone(nn.Module):

    def __init__(self):
        super(PitchBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 66)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class RollBone(nn.Module):

    def __init__(self):
        super(RollBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)
        layer3 = GlobalAvgPool2d()
        layer4 = FullyConnectLayer(512, 66)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class detectBone_multiScale_new(nn.Module):
    
    def __init__(self):
        super(detectBone_multiScale_new, self).__init__()

        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)  #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)  #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)  #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)

        self.avg_pool = GlobalAvgPool2d()

        self.fc = FullyConnectLayer(512, 5)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)
        detectRet = self.fc(multi_scale)

        return detectRet

class BinaryBone(nn.Module):
    def __init__(self):
        super(BinaryBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 2)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class WrinkleBone(nn.Module):
    def __init__(self):
        super(WrinkleBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 3)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class backBone_3channel(nn.Module):
    def __init__(self):
        super(backBone_3channel, self).__init__()

        layer0 = Conv2dBatchReLU(3, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class FaceAreaBone(nn.Module):
    def __init__(self):
        super(FaceAreaBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 6)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class detectBone_HeatMap(nn.Module):
    def __init__(self):

        super(detectBone_HeatMap, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1) #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2) #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2) #4
        layer3 = ConvTranspose2dBatchReLU(512,256,2,2,0)
        layer4 = ConvTranspose2dBatchReLU(256,128,2,2,0)
        layer5 = ConvTranspose2dBatchReLU(128,5,2,2,0)

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class FaceAreaBone_new(nn.Module):
    def __init__(self):
        super(FaceAreaBone_new, self).__init__()

        addcoords = AddCoords()

        layer0 = Conv2dBatchReLU(1+2, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64+2, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128+2, 128, 3, 2)    #16

        layer5 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer6 = Conv2dBatchReLU(128+2, 256, 3, 2)   #8
        layer7 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer8 = GlobalAvgPool2d()                 #1
        
        layer9 = FullyConnectLayer(512, 10)
        layer10 = FullyConnectLayer(10, 5)

        self.features = nn.Sequential(
            addcoords,
            layer0,
            layer1,
            addcoords,
            layer2,
            layer3,
            addcoords,
            layer4,
            layer5,
            addcoords,
            layer6,
            layer7,
            layer8
        )
        
        self.classifier = nn.Sequential(
            layer9,
            layer10
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class FaceAreaBone_five(nn.Module):
    def __init__(self):
        super(FaceAreaBone_five, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 5)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class angleBone(nn.Module):

    def __init__(self):
        super(angleBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 4)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class angleBone_six(nn.Module):

    def __init__(self):
        super(angleBone_six, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 6)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class FaceAreaBone_seven(nn.Module):
    def __init__(self):
        super(FaceAreaBone_seven, self).__init__()
        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        # layer4 = FullyConnectLayer(512, 14)
        # layer5 = FullyConnectLayer(14, 7)
        layer4 = FullyConnectLayer(512, 7)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            # layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x        
    
    def init_weight(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class FaceAreaBone_seven_new(nn.Module):
    def __init__(self):
        super(FaceAreaBone_seven_new, self).__init__()
        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        layer4 = FullyConnectLayer(512, 7)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x

class QualityCls(nn.Module):
    def __init__(self):
        super(QualityCls, self).__init__()
        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        layer4 = FullyConnectLayer(512, 2)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x
 
class RecognitionBone(nn.Module):

    def __init__(self,in_feat=512,num_classes=4185,cls_type='circleSoftmax',):
        super(RecognitionBone, self).__init__()

        self.in_feat = in_feat
        self.num_classes = num_classes
        self.cls_type = cls_type

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, self.in_feat, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        if cls_type == 'linear':          self.classifier = nn.Linear(self.in_feat, self.num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(self.in_feat, self.num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(self.in_feat, self.num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(self.in_feat, self.num_classes)
        else:
            raise KeyError("{cls_type} is invalid, please choose from "
                           "'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")

    def forward(self, x, targets=None):

        bn_feat = self.features(x)

        # test
        if None == targets:
            return bn_feat
        # train
        else:
            if self.classifier.__class__.__name__ == 'Linear':
                cls_outputs = self.classifier(bn_feat)
            else:
                cls_outputs = self.classifier(bn_feat, targets)

            pred_class_logits = F.linear(bn_feat, self.classifier.weight)

        return cls_outputs,pred_class_logits,bn_feat

class GazeBone(nn.Module):
    def __init__(self):
        super(GazeBone, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 3)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )

        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class PoseBone(nn.Module):
    def __init__(self):
        super(PoseBone, self).__init__()

        self.features = nn.Sequential(
            Conv2dBatchReLU(128, 128, 3, 1),  #16
            Conv2dBatchReLU(128, 256, 3, 2),  #8
            Conv2dBatchReLU(256, 256, 3, 1),  #8
            Conv2dBatchReLU(256, 512, 3, 2),  #4
            Conv2dBatchReLU(512, 1024, 1, 1), #4
            GlobalAvgPool2d(),                #1
        )

        self.dropout = nn.Dropout(0.5)

        self.yaw = FullyConnectLayer(1024, 120)
        self.pitch = FullyConnectLayer(1024, 66)
        self.roll = FullyConnectLayer(1024, 66)
            
    def forward(self, x):
    
        x = self.features(x)
        x = self.dropout(x)
        yaw = self.yaw(x)
        pitch = self.pitch(x)
        roll = self.roll(x)

        return yaw, pitch, roll

class GazeBone_5(nn.Module):

    def __init__(self):
        super(GazeBone_5, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 5)
        layer6 = nn.Sigmoid()

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )

        self.classifier = nn.Sequential(
            layer4,
            layer5,
            layer6,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class AngleBoneReg(nn.Module):
    def __init__(self):
        super(AngleBoneReg, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 1)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )

        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class FaceAreaBone_multiscale(nn.Module):
    def __init__(self):
        super(FaceAreaBone_multiscale, self).__init__()
        self.layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        self.layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        self.layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        self.layer3 = nn.Conv2d(512, 128, 4, 1, 0)
        
        self.avg_pool = GlobalAvgPool2d()
        self.classifier = FullyConnectLayer(512, 7) 

    def forward(self, x):
        x = self.layer0(x)
        out1 = self.avg_pool(x)

        x = self.layer1(x)
        out2 = self.avg_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        out3 = self.avg_pool(x)

        multi_scale = torch.cat([out1, out2, out3], 1)

        x = self.classifier(multi_scale)
        return x

class KLOccCls(nn.Module):
    def __init__(self):
        super(KLOccCls, self).__init__()
        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        layer4 = FullyConnectLayer(512, 2)
        # layer4 = FullyConnectLayer(512, 3)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x

class genderClassify(nn.Module):
    def __init__(self):
        super(genderClassify, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, 2)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class ageClassify(nn.Module):
    def __init__(self):
        super(ageClassify, self).__init__()

        layer0 = Conv2dBatchReLU(128, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1

        layer4 = FullyConnectLayer(512, 128)
        layer5 = FullyConnectLayer(128, 26)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )

        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ =='__main__':

    model = RecognitionBone()
    x = torch.randn(5, 128, 16, 16) 
    y = model(x)
    print(y.shape)
