# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-12-27 09:49:53
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-12-28 09:09:19
* @Description  : 性别数据集
'''
import copy
import inspect
import os
import os.path as osp
import random
import re
import sys
import time
import xml.dom.minidom
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import src.dataprocess.transform.augmentations as augmentations
import torch
from PIL import Image, ImageFilter
from src.dataprocess.transform.dataAug_box import randomAug_box
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm
import json

def getage(xmlFile):

    dom = xml.dom.minidom.parse(xmlFile)  
    root = dom.documentElement

    nBoxes = len(root.getElementsByTagName('age'))

    if(nBoxes != 1):
        print(xmlFile)
        sys.exit('{} is not right'.format(xmlFile))

    boxes = np.zeros((nBoxes,4))

    for iBox in range(nBoxes):

        itemlist = root.getElementsByTagName('age')
        age = int(float(itemlist[iBox].firstChild.data))

    return age

def getgender(xmlFile):

    dom = xml.dom.minidom.parse(xmlFile)  
    root = dom.documentElement

    nBoxes = len(root.getElementsByTagName('gender'))

    if(nBoxes != 1):
        print(xmlFile)
        sys.exit('{} is not right'.format(xmlFile))

    boxes = np.zeros((nBoxes,4))

    for iBox in range(nBoxes):

        itemlist = root.getElementsByTagName('gender')
        gender = int(float(itemlist[iBox].firstChild.data))

    return gender

def getboxes(xmlFile):

    if not os.path.exists(xmlFile):
        return np.zeros((1,4))
        
    dom = xml.dom.minidom.parse(xmlFile)  
    root = dom.documentElement

    nBoxes = len(root.getElementsByTagName('xmin'))

    boxes = np.zeros((nBoxes,4))

    for iBox in range(nBoxes):

        itemlist = root.getElementsByTagName('xmin')
        minX = int(float(itemlist[iBox].firstChild.data))

        itemlist = root.getElementsByTagName('ymin')
        minY = int(float(itemlist[iBox].firstChild.data))

        itemlist = root.getElementsByTagName('xmax')
        maxX = int(float(itemlist[iBox].firstChild.data))

        itemlist = root.getElementsByTagName('ymax')
        maxY = int(float(itemlist[iBox].firstChild.data))

        boxes[iBox][0] = minX
        boxes[iBox][1] = minY
        boxes[iBox][2] = maxX
        boxes[iBox][3] = maxY

    return boxes

def getbox(json_file):
    if json_file.endswith('.json'):
        with open(json_file,'rb') as f:
            data = json.load(f)
        points = data['shapes'][0]['points']
        x,y,w,h = points[0],points[1],points[2]-points[0],points[3]-points[1]
    elif json_file.endswith('.xml'):
        boxes = getboxes(json_file)
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2]-box[0],box[3]-box[1]
    else:
        print(json_file)
        sys.exit()
    return [x,y,w,h]

def randomAug_boxV2(img,box,scale):

    height, width = img.shape[0:2]

    x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
    w = x2 - x1
    h = y2 - y1

    if w < 20 or h < 20:
        return (False,'w or h is very small')

    if random.random() < 0.5:
        delta_x1 = np.random.randint(0,int(w * scale))
        delta_y1 = np.random.randint(0,int(h * scale))
        delta_x2 = np.random.randint(0,int(w * scale)) 
        delta_y2 = np.random.randint(0,int(h * scale)) 
    else:
        delta_x1 = np.random.randint(int(w * scale), int(w * scale * 2))
        delta_y1 = np.random.randint(int(h * scale), int(h * scale * 2))
        delta_x2 = np.random.randint(int(w * scale), int(w * scale * 2)) 
        delta_y2 = np.random.randint(int(h * scale), int(h * scale * 2)) 

    nx1 = max(x1 - delta_x1,0)
    ny1 = max(y1 - delta_y1,0)
    nx2 = min(x2 + delta_x2,width)
    ny2 = min(y2 + delta_y2,height)

    if (ny2 < ny1 + 20) or (nx2 < nx1 + 20):
        return (False,'ny2 or nx2 is very small')

    # 将点归一化到裁剪区域中
    x1 = (x1 - nx1) * 128 / (nx2 - nx1)
    y1 = (y1 - ny1) * 128 / (ny2 - ny1)

    x1 = x1 / 128.0000000000
    y1 = y1 / 128.0000000000

    x2 = (x2 - nx1) * 128 / (nx2 - nx1)
    y2 = (y2 - ny1) * 128 / (ny2 - ny1)

    x2 = x2 / 128.0000000000
    y2 = y2 / 128.0000000000

    cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return (True, cropped_im, [x1,y1,x2,y2])

def aug(image, preprocess,all_ops=True):
    """Perform AugMix augmentations and compute mixture.

    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(3):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, 3)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix

    return mixed

class DatasetGesture(torchDataset):

    def __init__(self, imgDir, imgTxt, size=128, imgChannel=1, isTrain='train'):

        self.size = size

        self.imgTxt = os.path.join(imgDir,imgTxt)
        with open(self.imgTxt,'r') as f:
            lines = f.readlines()

        # get imgpath
        self.imgPathList = []
        self.xmlPathList = []
        self.labelList = []
        for line in tqdm(lines):
            imgFile = imgDir + line.strip().split(' ')[0]
            xmlFile = osp.splitext(imgFile)[0] + '.xml'
            jsonFile = osp.splitext(imgFile)[0] + '.json'
            label = int(line.strip().split(' ')[1])
            self.imgPathList.append(imgFile)
            if osp.exists(xmlFile):
                self.xmlPathList.append(xmlFile)
            if osp.exists(jsonFile):
                self.xmlPathList.append(jsonFile)
            self.labelList.append(label)

        assert len(self.imgPathList) == len(self.xmlPathList)

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.xmlPathList):',len(self.xmlPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self,index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.xmlPathList[index]) #[x y w h]
        gesture = self.labelList[index]
        

        # get new img and new box
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        box = [box] #[[x1,y1,x2,y2]]
        img_new, box_new = randomAug_box(img,box)
        ret = randomAug_boxV2(img_new,box_new,0.15)

        if(ret[0] == False):
            print('box_ori:',box_ori)
            print('box:',box)
            print('box_new:',box_new)
            sys.exit('{} have problem:{}'.format(self.imgPathList[index],ret[1]))
        else:
            cropped_im = ret[1]

        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return resized_im, torch.FloatTensor([gesture]).type(torch.FloatTensor)

class DatasetGestureSim(DatasetGesture):
    '''
    return ori PIL image for augmix
    '''
    def __init__(self, imgDir, imgTxt, size=128, imgChannel=1, isTrain='train'):
        super(DatasetGestureSim, self).__init__(imgDir, imgTxt, size=size,isTrain=isTrain)

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self,index):

        img = cv2.imread(self.imgPathList[index],1)
        box = getbox(self.xmlPathList[index])
        gesture = self.labelList[index]

        # get new img and new box
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        box = [box] #[[x1,y1,x2,y2]]
        img_new, box_new = randomAug_box(img,box)
        ret = randomAug_boxV2(img_new,box_new,0.15)

        if(ret[0] == False):
            sys.exit('{} have problem:{}'.format(self.imgPathList[index],ret[1]))
        else:
            cropped_im = ret[1]

        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(resized_im,cv2.COLOR_BGR2RGB))

        return image, torch.FloatTensor([gesture]).type(torch.FloatTensor)

def preprocess(imagePil):

    imagePil = imagePil.convert("L")
    imageNp = np.asarray(imagePil)
    imageNp = imageNp[:,:,np.newaxis] # 128 128 1
    imageTensor = F.to_tensor(imageNp)

    return imageTensor

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess=preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i] 
    # x is PIL Image shape is 128 128 3

    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)
