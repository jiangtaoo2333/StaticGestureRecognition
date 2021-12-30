import argparse
import os
import os.path as osp
import sys
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
import torch.nn as nn
import cv2
dirpath = osp.dirname(osp.dirname(osp.abspath(__file__))).replace('\\','/')
sys.path.append(dirpath)
import timm


def get_args():

    parser = argparse.ArgumentParser("MultiTaskOnFace build by Jiangtao")

    parser.add_argument('--config', 
                        default='{}/configs/gesture/dms_easyNet_crossentroy_cosineannealing_augmix.py'.format(dirpath),help='train config file path')

    args = parser.parse_args()

    return args

args = get_args()
cfg = Config.fromfile(args.config)

class StaticGesture():

    def __init__(self,
                cfg=cfg,
                checkpoint='easyNet_DMS_gender_best_0.967529296875.pkl'):

        self.cfg = cfg
        self.model = timm.create_model(self.cfg.modelName, pretrained=False, num_classes=self.cfg.numClasses,
                                in_chans=self.cfg.channels).cuda()


        filename = self.cfg.filename
        basefilename = osp.basename(filename)
        basefilename = osp.splitext(basefilename)[0]
        self.modelPath = osp.join('{}/work_dirs/'.format(dirpath),basefilename)
        self.modelPath = osp.join(self.modelPath,checkpoint)
        print('self.modelPath:',self.modelPath)

        self.model.load_state_dict(torch.load(self.modelPath),strict=False)
        self.model.cuda().eval()

    def classify(self,image,box):
        '''
        image is numpy h w
        box is [x,y,x,y]
        '''
        scale = 0.10
        xmin,ymin,xmax,ymax = box
        roiw = xmax - xmin
        roih = ymax - ymin

        xmin -= roiw * scale
        xmax += roiw * scale
        ymin -= roih * scale
        ymax += roih * scale

        xmin = np.clip(xmin,0,image.shape[1]-1)
        xmax = np.clip(xmax,0,image.shape[1]-1)
        ymin = np.clip(ymin,0,image.shape[0]-1)
        ymax = np.clip(ymax,0,image.shape[0]-1)

        x1 = int(xmin)
        x2 = int(xmax)
        y1 = int(ymin)
        y2 = int(ymax)

        img = image[y1:y2,x1:x2]

        # 输入图片预处理
        img = cv2.resize(img, (self.cfg.imgSize,self.cfg.imgSize), interpolation = cv2.INTER_CUBIC)*0.0039216
        img = img[np.newaxis] # 1 128 128
        img_ = torch.from_numpy(img) # 1 128 128
        img_ = img_.unsqueeze_(0) # 1 1 128 128

        img_ = img_.cuda()

        pre_ = self.model(img_.float())
        m = nn.Softmax(dim=1)
        pre_ = m(pre_)
        pre_ = pre_.cpu().detach().numpy().reshape((1,-1))
        res = np.argmax(pre_,axis=-1)

        if res[0] == 0:
            label = 'palm'
        if res[0] == 1:
            label = 'singleFinger'
        if res[0] == 2:
            label = 'doubleFinger'

        score = pre_[0][res[0]]

        return label,score


if __name__ == '__main__':

    SataticGestureCls = StaticGesture()

    img = cv2.imread('./demo/images/1.jpg',0)
    box = [1057,504,1207,706]
    x1,y1,x2,y2 = box
    label,score = SataticGestureCls.classify(img,box)
    print(label)
    print(score)
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.imshow('img',img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

