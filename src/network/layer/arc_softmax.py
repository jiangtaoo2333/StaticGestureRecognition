'''
Author: Jiangtao
Date: 2020-08-29 14:39:39
LastEditors: Jiangtao
LastEditTime: 2020-08-29 16:42:22
Description: 
'''
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcSoftmax(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = 128   #scale
        self._m = 0.15  #margin

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        # 处理logit边界，logit即为正确预测的结果
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        
        # 处理hard样本，就是比预测值比logits大的样本
        mask = cos_theta > cos_theta_m
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)

        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
