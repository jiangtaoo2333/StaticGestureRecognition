import os
from collections import OrderedDict

import timm.models.layers.layers_by_jiangtao as layer
import torch
import torch.nn as nn

from .registry import register_model


@register_model
class easyNet(nn.Module):
    
    def __init__(self,pretrained=False,num_classes=3,in_chans=1):
        super(easyNet, self).__init__()

        self.baseBone = layer.backBone()
        self.gestureBone = layer.gestureClassify(num_classes)

    def forward(self,x):
        
        x = self.baseBone(x)
        x = self.gestureBone(x)

        return x
