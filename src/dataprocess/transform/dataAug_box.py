'''
@Author: Jiangtao
@Date: 2019-07-24 11:31:08
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-10-08 14:56:20
@Description: 
'''
import copy
import math
import os
import pickle
import random
import time
import xml.dom.minidom
import torch

import numpy as np

import cv2
import copy
from torchvision import transforms
from .random_erasing import RandomErasing

# 根据box进行合理放大
def randomeResize(img,box,isDeformity=False):

    height, width = img.shape[0:2]

    minX = box[0][0]
    minY = box[0][1]
    maxX = box[0][2]
    maxY = box[0][3]

    w = maxX - minX
    h = maxY - minY

    if not isDeformity:

        if random.random() < 0.75:
            delta_x1 = np.random.randint(int(w * 0.0), int(w * 0.75))
            delta_y1 = np.random.randint(int(h * 0.0), int(h * 0.75))
            delta_x2 = np.random.randint(int(w * 0.0), int(w * 0.75)) 
            delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.75)) 
        else:
            delta_x1 = np.random.randint(int(w * 0.75), int(w * 1.5))
            delta_y1 = np.random.randint(int(h * 0.75), int(h * 1.5))
            delta_x2 = np.random.randint(int(w * 0.75), int(w * 1.5)) 
            delta_y2 = np.random.randint(int(h * 0.75), int(h * 1.5)) 
        
    else:
        # 正常外扩
        if random.random() < 0.75:

            if random.random() < 0.75:
                delta_x1 = np.random.randint(int(w * 0.0), int(w * 0.75))
                delta_y1 = np.random.randint(int(h * 0.0), int(h * 0.75))
                delta_x2 = np.random.randint(int(w * 0.0), int(w * 0.75)) 
                delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.75)) 
            else:
                delta_x1 = np.random.randint(int(w * 0.75), int(w * 1.5))
                delta_y1 = np.random.randint(int(h * 0.75), int(h * 1.5))
                delta_x2 = np.random.randint(int(w * 0.75), int(w * 1.5)) 
                delta_y2 = np.random.randint(int(h * 0.75), int(h * 1.5)) 
        
        # 上下左右四种残缺
        else:
            dice = random.random()
            if dice < 0.25:
                delta_x1 = -np.random.randint(int(w * 0.0), int(w * 0.5))
                delta_y1 = np.random.randint(int(h * 0.0), int(h * 0.75))
                delta_x2 = np.random.randint(int(w * 0.0), int(w * 0.75)) 
                delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.75))
            elif dice >=0.25 and dice < 0.5:
                delta_x1 = np.random.randint(int(w * 0.0), int(w * 0.75))
                delta_y1 = -np.random.randint(int(h * 0.0), int(h * 0.5))
                delta_x2 = np.random.randint(int(w * 0.0), int(w * 0.75)) 
                delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.75))
            elif dice >=0.5 and dice < 0.75:
                delta_x1 = np.random.randint(int(w * 0.0), int(w * 0.75))
                delta_y1 = np.random.randint(int(h * 0.0), int(h * 0.75))
                delta_x2 = -np.random.randint(int(w * 0.0), int(w * 0.5)) 
                delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.75))
            else:
                delta_x1 = np.random.randint(int(w * 0.0), int(w * 0.75))
                delta_y1 = np.random.randint(int(h * 0.0), int(h * 0.75))
                delta_x2 = np.random.randint(int(w * 0.0), int(w * 0.75)) 
                delta_y2 = -np.random.randint(int(h * 0.0), int(h * 0.5))

    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x2,width)
    ny2 = min(maxY + delta_y2,height)

    box[0][0] -= int(nx1)
    box[0][2] -= int(nx1)
    box[0][0] = max(box[0][0],0)
    box[0][2] = max(box[0][2],0)

    box[0][1] -= int(ny1)
    box[0][3] -= int(ny1)
    box[0][1] = max(box[0][1],0)
    box[0][3] = max(box[0][3],0)

    # print('box',box)
    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,box

# 根据box进行合理放大
def randomeResizeGaze(img,box):

    height, width = img.shape[0:2]

    minX = box[0][0]
    minY = box[0][1]
    maxX = box[0][2]
    maxY = box[0][3]

    boxOri = copy.deepcopy(box)

    w = maxX - minX
    h = maxY - minY

    delta_x1 = np.random.randint(int(w * 0.10), int(w * 0.35))
    delta_y1 = np.random.randint(int(w * 0.10), int(w * 0.35))
    delta_x2 = np.random.randint(int(w * 0.10), int(w * 0.35)) 
    delta_y2 = np.random.randint(int(w * 0.10), int(w * 0.35)) 

    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x2,width)
    ny2 = min(maxY + delta_y2,height)

    # assert ny2 > ny1,"boxOri:{},box:{},w:{},h:{},delta_y1:{},delta_y2:{}".format(boxOri,box,w,h,delta_y1,delta_y2)
    # assert nx2 > nx1,"boxOri:{},box:{},w:{},h:{},delta_x1:{},delta_x2:{}".format(boxOri,box,w,h,delta_x1,delta_x2)

    box[0][0] -= int(nx1)
    box[0][2] -= int(nx1)
    box[0][1] -= int(ny1)
    box[0][3] -= int(ny1)

    # assert box[0][3] > box[0][1],"boxOri:{},box:{},w:{},h:{},delta_y1:{},delta_y2:{}".format(boxOri,box,w,h,delta_y1,delta_y2)
    # assert box[0][2] > box[0][0],"boxOri:{},box:{},w:{},h:{},delta_x1:{},delta_x2:{}".format(boxOri,box,w,h,delta_x1,delta_x2)

    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,box

# 根据box进行合理放大
def randomeResizeOnet(img,box):

    height, width = img.shape[0:2]

    minX = box[0][0]
    minY = box[0][1]
    maxX = box[0][2]
    maxY = box[0][3]

    boxOri = copy.deepcopy(box)

    w = maxX - minX
    h = maxY - minY

    # delta_x1 = np.random.randint(50, 80)
    # delta_y1 = np.random.randint(int(w * 0.10), int(w * 0.65))
    # delta_x2 = np.random.randint(50, 80) 
    # delta_y2 = np.random.randint(int(w * 0.10), int(w * 0.65)) 

    # delta_x1 = 50
    # delta_x2 = 50
    # delta_y1 = int(w * 0.20)
    # delta_y2 = int(w * 0.20)

    delta_x1 = 20
    delta_x2 = 20
    delta_y1 = 20
    delta_y2 = 20


    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x2,width)
    ny2 = min(maxY + delta_y2,height)

    # assert ny2 > ny1,"boxOri:{},box:{},w:{},h:{},delta_y1:{},delta_y2:{}".format(boxOri,box,w,h,delta_y1,delta_y2)
    # assert nx2 > nx1,"boxOri:{},box:{},w:{},h:{},delta_x1:{},delta_x2:{}".format(boxOri,box,w,h,delta_x1,delta_x2)

    box[0][0] -= int(nx1)
    box[0][2] -= int(nx1)
    box[0][1] -= int(ny1)
    box[0][3] -= int(ny1)

    # assert box[0][3] > box[0][1],"boxOri:{},box:{},w:{},h:{},delta_y1:{},delta_y2:{}".format(boxOri,box,w,h,delta_y1,delta_y2)
    # assert box[0][2] > box[0][0],"boxOri:{},box:{},w:{},h:{},delta_x1:{},delta_x2:{}".format(boxOri,box,w,h,delta_x1,delta_x2)

    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,box

# 随机旋转+-30度
def randomeRotate(img,box):

    pts = box.reshape((2,2))
    num = 15
    angle = np.random.randint(-num,num)
    rad = angle * np.pi / 180.0

    height ,width = img.shape[0:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height))

    pts_ = np.zeros_like(pts)
    pts[:,1] = height - pts[:,1]
    pts_[:,0] = (pts[:,0] - (width / 2)) * np.cos(rad) - (pts[:,1] - (height / 2)) * np.sin(rad) + (width / 2)
    pts_[:,1] = (pts[:,1] - (height / 2)) * np.cos(rad) + (pts[:,0] - (width / 2)) * np.sin(rad) + (height / 2)
    pts_[:,1] = height - pts_[:,1]
    
    box = pts_.reshape((1,4))

    return rotated,box

# 随机对称
def randomFlip(img,box):

    height ,width = img.shape[0:2]
    flipped_img = cv2.flip(img,1)

    minX = box[0][0]
    minY = box[0][1]
    maxX = box[0][2]
    maxY = box[0][3]

    box[0][0] = width - maxX
    box[0][2] = width - minX

    return flipped_img,box

# 随机偏移
def randomTranslation(img,box):

    pts = box.reshape((2,2))
    height,width = img.shape[0:2]
    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)
    w = min(minX,width - maxX)
    h = min(minY,height - maxY)
    # print(w,h)
    if random.choice([True,False]):
        w = -w
    if random.choice([True,False]):
        h = -h
    if w > 0:          
        w = random.randint(0, int(w))
    else:               
        w = random.randint(int(w), 0)               
    if h > 0:
        h = random.randint(0, int(h))         
    else:
        h = random.randint(int(h), 0)

    affine = np.float32([[1,0,w],[0,1,h]])
    img = cv2.warpAffine(img,affine,(img.shape[1],img.shape[0]))
    pts += np.array([w,h])
    box = pts.reshape((1,4))

    return img,box

# 随机HSV空间变化
def randomHSV(img,box):

    hue_vari = 10
    sat_vari = 0.2
    val_vari = 0.2

    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    if len(img.shape) == 2:
        img = cv2.merge([img,img,img])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    img = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img,box

# 随机滤波
def randomBlur(img,box):

    n = random.random()
    if n<=0.25:
        img_mean = cv2.blur(img, (5,5))
        return img_mean,box
    if n <= 0.5:
        img_Guassian = cv2.GaussianBlur(img,(5,5),0)
        return img_Guassian,box
    if n <= 0.75:
        img_median = cv2.medianBlur(img, 5)
        return img_median,box
    if n <= 1:
        img_bilater = cv2.bilateralFilter(img,9,75,75)

        return img_bilater,box

# 随机噪声
def randomNoise(img,box):

    N = int(img.shape[0] * img.shape[1] * 0.001)
    for i in range(N): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255

    return img,box

# 随机伪逆光
def randomLogTransform(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    H,W = img.shape[0:2]

    x = random.choice([17,18,19])
    y = random.choice([2,3])

    for i in range(H):
        for j in range(W):
            img[i][j] = x*math.log(1+img[i][j],y)

    return img

# 随机擦除
def randomErasing(img):

    RandomErasing_OB = RandomErasing(probability=0.5, sl=0.02, sh=0.1,device='cpu')
    # transform = transforms.Compose([RandomErasing,])

    # print('0',img.shape)
    img = img[np.newaxis]
    # print('a',img.shape)
    img = torch.from_numpy(img).clone()
    # print('b',img.shape)
    # img = img.transpose(0,2)
    # print('c',img.shape)
    img = RandomErasing_OB(img)
    # print('d',img.shape)
    # img = img.transpose(0,2)
    # print('e',img.shape)
    img = img.numpy()
    img = img.squeeze()
    # print('e',img.shape)

    return img

# 集成随机
def randomAug_box(img,box):

    img,box = randomeResize(img,box)

    # if random.random() > 0.5:
    #     img,box = randomFlip(img,box)
        
    if(1):
        img = randomErasing(img)

        if random.random() > 0.5:
            img,box = randomFlip(img,box)

        if random.random() > 0.5:
            img,box = randomHSV(img,box)

        # if random.random() > 0.5:
        #     img,box = randomBlur(img,box)

        # if random.random() > 0.5:
        #     img,box = randomNoise(img,box)

    return img,box

# 集成随机
def randomAugOnet_box(img,box):

    img,box = randomeResizeOnet(img,box)

    if(1):
        # img = randomErasing(img)

        # if random.random() > 0.5:
        #     img,box = randomFlip(img,box)

        if random.random() > 0.5:
            img,box = randomHSV(img,box)

    return img,box

def randomAug_Gaze(img,box):

    img,box = randomeResizeGaze(img,box)

    return img,box


if __name__ == '__main__':

    while True:

        imgFile = os.path.join('./dataset/train/emotion/DSM/check/neutral/normal/','0eb39fd2-273f-11ea-8d44-b42e994b717c50.jpg')
        xmlFile = os.path.join('./dataset/train/emotion/DSM/check/neutral/normal/','0eb39fd2-273f-11ea-8d44-b42e994b717c50.xml')

        img = cv2.imread(imgFile,-1)

        dom = xml.dom.minidom.parse(xmlFile)  
        root = dom.documentElement

        itemlist = root.getElementsByTagName('xmin')
        minX = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('ymin')
        minY = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('xmax')
        maxX = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('ymax')
        maxY = int(float(itemlist[0].firstChild.data))

        boxes = np.zeros((1,4))
        boxes[0][0] = minX
        boxes[0][1] = minY
        boxes[0][2] = maxX
        boxes[0][3] = maxY

        print(boxes)
        print(img.shape)

        img,boxes = randomAug(img,boxes)

        print(boxes)
        print(img.shape)

        minX = boxes[0][0]
        minY = boxes[0][1]
        maxX = boxes[0][2]
        maxY = boxes[0][3]

        cv2.rectangle(img,(int(minX),int(minY)),(int(maxX),int(maxY)),(255,0,0),1)

        cv2.imshow('img',img)
        cv2.waitKey(0)