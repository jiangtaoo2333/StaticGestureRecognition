'''
@Author: Jiangtao
@Date: 2019-07-24 11:31:08
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-08-17 09:33:41
@Description: 
'''
import copy
import math
import os
import pickle
import random
import time
import xml.dom.minidom
from PIL import Image
import torch

import cv2
import numpy as np
from torchvision import transforms
from .random_erasing import RandomErasing
import sys

def getbox(xmlFile):

    if not os.path.exists(xmlFile):
        return np.zeros((1,4))
        
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

    return boxes

def getpoints(ptsFile):

    pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1).reshape((-1,2))
    
    if 68 == pts.shape[0]:
        
        pts = pts.reshape((68,1,2))
        
        eyePts = np.concatenate((pts[0,:],pts[36,:],pts[37,:],pts[39,:],
                                pts[41,:],pts[27,:],pts[42,:],pts[44,:],
                                pts[45,:],pts[46,:],pts[16,:]),axis=0)

        mouthPts = np.concatenate((pts[8,:],pts[30,:],pts[48,:],pts[51,:],
                                pts[54,:],pts[57,:],pts[62,:],pts[66,:],),axis=0)

        pts = np.concatenate((eyePts,mouthPts),axis=0) 

    elif 19 == pts.shape[0]:
        pass

    assert pts.shape == (19,2)

    return pts

# logTransform
def randomLogTransform(img,pts):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    H,W = img.shape[0:2]

    x = random.choice([17,18,19])
    y = random.choice([2,3])

    for i in range(H):
        for j in range(W):
            img[i][j] = x*math.log(1+img[i][j],y)

    return img,pts

# 根据box进行合理放大
def randomeResize(img,pts):

    height, width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    w = maxX - minX
    h = maxY - minY
    # 避免特别小的人脸，对后面的扩增有影响
    h = max(h,150)
    
    # enlarge to whole head
    # minY -= 0.3 * h
    # h = 1.3*h

    first = 0
    second = 0

    # 正常外扩
    if(random.random() <= 0.5):
        first = 0
        if random.random() < 0.60: #小外扩
            second = 0
            delta_x1 = np.random.randint(int(w * 0.), int(w * 0.20))
            delta_y1 = np.random.randint(int(h * 0.), int(h * 0.20))
            delta_x2 = np.random.randint(int(w * 0.), int(w * 0.20)) 
            delta_y2 = np.random.randint(int(h * 0.), int(h * 0.20)) 
        else: #大外扩
            second = 1
            delta_x1 = np.random.randint(int(w * 0.05), int(w * 0.40))
            delta_y1 = np.random.randint(int(h * 0.05), int(h * 0.40))
            delta_x2 = np.random.randint(int(w * 0.05), int(w * 0.40)) 
            delta_y2 = np.random.randint(int(h * 0.05), int(h * 0.40)) 

        nx1 = int(max(minX - delta_x1,0))
        ny1 = int(max(minY - delta_y1,0))
        nx2 = int(min(maxX + delta_x2,width))
        ny2 = int(min(maxY + delta_y2,height))

    # 上下裁剪式外扩
    else:
        first = 1
        delta_x1 = np.random.randint(int(w * 0.20), int(w * 0.35))
        delta_y1 = np.random.randint(int(h * 0.45), int(h * 0.5))
        delta_x2 = np.random.randint(int(w * 0.20), int(w * 0.35)) 
        delta_y2 = np.random.randint(int(h * 0.20), int(h * 0.35)) 

        eyeMinY = int(pts[0:11].min(axis=0)[1])
        headMinY = max(eyeMinY - 0.5*h,0)

        eyeMaxY = int(pts[0:11].max(axis=0)[1])
        headMaxY = min(eyeMaxY + 1.0*h,height)
        headMaxYSecond = min(eyeMaxY + 0.3*h,height)

        # 上下不可能同时残缺，所以两种残缺分别考虑
        if(random.random() < 0.5):
            second = 0
            nx1 = int(max(minX - delta_x1,0))
            if(eyeMinY > headMinY):
                ny1 = np.random.randint(headMinY, eyeMinY)
            else:
                ny1 = eyeMinY
            nx2 = int(min(maxX + delta_x2,width))
            ny2 = int(min(maxY + delta_y2,height))
        else:
            second = 1
            nx1 = int(max(minX - delta_x1,0))
            ny1 = int(max(minY - delta_y1,0))
            nx2 = int(min(maxX + delta_x2,width))
            if(headMaxY > headMaxYSecond):
                ny2 = np.random.randint(headMaxYSecond, headMaxY)
            else:
                ny2 = headMaxY

    assert ny2 > ny1,'wrong in data {} {} {} {} {} {}'.format(first,second,nx1,nx2,ny1,ny2)
    assert nx2 > nx1,'wrong in data {} {} {} {} {} {}'.format(first,second,nx1,nx2,ny1,ny2)

    pts[:,0] -= nx1
    pts[:,1] -= ny1

    if len(img.shape) == 3:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts

def randomeOnetResize(img,pts):

    height, width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    w = maxX - minX
    h = maxY - minY
    # if(h<=10):
    #     h = 50
    # enlarge to whole head
    # minY -= 0.3 * h
    # h = 1.3*h

    dice = random.random()

    # if dice < 0.5: #小外扩
    #     delta_x1 = np.random.randint(int(w * 0.20), int(w * 0.40))
    #     delta_y1 = np.random.randint(int(h * 0.20), int(h * 0.40))
    #     delta_x2 = np.random.randint(int(w * 0.20), int(w * 0.40)) 
    #     delta_y2 = np.random.randint(int(h * 0.20), int(h * 0.40)) 
    # else:
    #     delta_x1 = np.random.randint(int(w * 0.40), int(w * 0.50))
    #     delta_y1 = np.random.randint(int(h * 0.40), int(h * 0.50))
    #     delta_x2 = np.random.randint(int(w * 0.40), int(w * 0.50)) 
    #     delta_y2 = np.random.randint(int(h * 0.40), int(h * 0.50)) 

    delta_x1 = np.random.randint(50, 80)
    delta_x2 = np.random.randint(50, 80)
    delta_y1 = np.random.randint(int(w * 0.00), int(w * 0.40))
    delta_y2 = np.random.randint(int(w * 0.15), int(w * 0.40))

    nx1 = int(max(minX - delta_x1,0))
    ny1 = int(max(minY - delta_y1,0))
    nx2 = int(min(maxX + delta_x2,width))
    ny2 = int(min(maxY + delta_y2,height))
    
    pts[:,0] -= nx1
    pts[:,1] -= ny1

    if len(img.shape) == 3:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts


def randomeResizeEye(img,pts):

    height, width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    w = maxX - minX
    h = maxY - minY
    # enlarge to whole head
    # minY -= 0.3 * h
    # h = 1.3*h

    dice = random.random()

    #小外扩
    if dice < 0.50:
        delta_x1 = np.random.randint(int(w * 0.), int(w * 0.10))
        delta_y1 = np.random.randint(int(h * 0.), int(h * 0.10))
        delta_x2 = np.random.randint(int(w * 0.), int(w * 0.10)) 
        delta_y2 = np.random.randint(int(h * 0.), int(h * 0.10))
    #大外扩
    else:
        delta_x1 = np.random.randint(int(w * 0.10), int(w * 0.20))
        delta_y1 = np.random.randint(int(h * 0.10), int(h * 0.20))
        delta_x2 = np.random.randint(int(w * 0.10), int(w * 0.20))
        delta_y2 = np.random.randint(int(h * 0.10), int(h * 0.20))
    # elif (dice>=0.50) and (dice<0.62): #下巴残缺场景
    #     delta_x1 = np.random.randint(int(w * 0.), int(w * 0.15))
    #     delta_y1 = np.random.randint(int(h * 0.), int(h * 0.15))
    #     delta_x2 = np.random.randint(int(w * 0.), int(w * 0.15)) 
    #     delta_y2 = -np.random.randint(int(h * 0.), int(h * 0.15)) 
    # elif (dice>=0.62) and (dice<0.75): # 额头残缺
    #     delta_x1 = np.random.randint(int(w * 0.), int(w * 0.15))
    #     delta_y1 = -np.random.randint(int(h * 0.), int(h * 0.15))
    #     delta_x2 = np.random.randint(int(w * 0.), int(w * 0.15)) 
    #     delta_y2 = np.random.randint(int(h * 0.), int(h * 0.15)) 
    # elif dice>=0.75:  #左右贴边
    #     if random.random() > 0.5:
    #         delta_x1 = np.random.randint(int(w * 0.), int(w * 0.10))
    #         delta_y1 = np.random.randint(int(h * 0.), int(h * 0.10))
    #         delta_x2 = np.random.randint(int(w * 0.), int(w * 0.10)) 
    #         delta_y2 = np.random.randint(int(h * 0.), int(h * 0.10)) 
    #     else:
    #         delta_x1 = np.random.randint(int(w * 0.), int(w * 0.10))
    #         delta_y1 = np.random.randint(int(h * 0.), int(h * 0.10))
    #         delta_x2 = np.random.randint(int(w * 0.), int(w * 0.10)) 
    #         delta_y2 = np.random.randint(int(h * 0.), int(h * 0.10))          

    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x2,width)
    ny2 = min(maxY + delta_y2,height)

    pts[:,0] -= nx1
    pts[:,1] -= ny1

    if len(img.shape) == 3:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts


def randomeResizeMouth(img,pts):

    height, width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    w = maxX - minX
    h = maxY - minY
    # enlarge to whole head
    minY -= 0.3 * h
    h = 1.3*h

    dice = random.random()

    if dice < 0.33: #小外扩
        delta_x1 = np.random.randint(int(w * 0.), int(w * 0.10))
        delta_y1 = np.random.randint(int(h * 0.), int(h * 0.10))
        delta_x2 = np.random.randint(int(w * 0.), int(w * 0.10)) 
        delta_y2 = np.random.randint(int(h * 0.), int(h * 0.10)) 
    elif (dice>=0.33) and (dice<0.66): #大外扩
        delta_x1 = np.random.randint(int(w * 0.10), int(w * 0.20))
        delta_y1 = np.random.randint(int(h * 0.10), int(h * 0.20))
        delta_x2 = np.random.randint(int(w * 0.10), int(w * 0.20)) 
        delta_y2 = np.random.randint(int(h * 0.10), int(h * 0.20))     
    elif (dice>=0.66): # 额头残缺
        delta_x1 = np.random.randint(int(w * 0.), int(w * 0.15))
        delta_y1 = -np.random.randint(int(h * 0.), int(h * 0.15))
        delta_x2 = np.random.randint(int(w * 0.), int(w * 0.15)) 
        delta_y2 = np.random.randint(int(h * 0.), int(h * 0.15)) 

    nx1 = max(minX - delta_x1,0)
    ny1 = max(minY - delta_y1,0)
    nx2 = min(maxX + delta_x2,width)
    ny2 = min(maxY + delta_y2,height)
    
    pts[:,0] -= nx1
    pts[:,1] -= ny1

    if len(img.shape) == 3:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts

# 随机裁剪
def randomeCrop(img,pts):

    height ,width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    x1,y1,x2,y2 = minX,minY,maxX,maxY

    w = x2 - x1
    h = y2 - y1


    if w*0.5 <=0 or h*0.5<=0:
        return img,pts
        
    # flag =  random.random()

    # if flag >= 0.5:
    #     delta_x1 = np.random.randint(0,int(w * 0.5))
    #     delta_y1 = np.random.randint(0,int(h * 0.5))
    #     delta_x2 = np.random.randint(0,int(w * 0.5)) 
    #     delta_y2 = np.random.randint(0,int(h * 0.5)) 
    # else:
    delta_x1 = np.random.randint(int(w * 0.5), int(w * 1))
    delta_y1 = np.random.randint(int(h * 0.5), int(h * 1))
    delta_x2 = np.random.randint(int(w * 0.5), int(w * 1)) 
    delta_y2 = np.random.randint(int(h * 0.5), int(h * 1))


    nx1 = max(x1 - delta_x1,0)
    ny1 = max(y1 - delta_y1,0)
    nx2 = min(x2 + delta_x2,width)
    ny2 = min(y2 + delta_y2,height)
    
    pts[:,0] -= nx1
    pts[:,1] -= ny1

    if len(img.shape) >2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
    if len(img.shape) == 2:
        img = img[int(ny1): int(ny2), int(nx1): int(nx2)]

    return img,pts

# 随机旋转+-5度
def randomeRotate(img,pts):

    num = 5
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
    
    return rotated,pts_

# 随机对称
def randomFlip(img,pts):

    height, width = img.shape[0:2]
    flipped_img = cv2.flip(img,1)
    pts_ = np.zeros_like(pts)
    pts_[:,1] = pts[:,1]
    pts_[:,0] = width - pts[:,0]
    pts = pts_

    if 19 == pts.shape[0]:
        for i,j in [[0,10],[1,8],[2,7],[3,6],[4,9],[13,15]]:
            temp = copy.deepcopy(pts[i])
            pts[i] = pts[j]
            pts[j] = temp

    elif 11 == pts.shape[0]:
        for i,j in [[0,10],[1,8],[2,7],[3,6],[4,9]]:
            temp = copy.deepcopy(pts[i])
            pts[i] = pts[j]
            pts[j] = temp

    elif 21 == pts.shape[0]:
        for i,j in [[0,12],[1,9],[2,8],[3,7],[4,10],[5,11],[15,17]]:
            temp = copy.deepcopy(pts[i])
            pts[i] = pts[j]
            pts[j] = temp
    
    else:
        try:
            print(pts.shape)
            sys.exit('pts is wrong')
        except SystemExit as e:
            print(repr(e))
            os.exit()


    return flipped_img,pts

# 随机偏移
def randomTranslation(img,pts):

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

    return img,pts

# 随机HSV空间变化
def randomHSV(img,pts):

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
    
    return img,pts

# 随机滤波
def randomBlur(img,pts):

    img_mean = cv2.blur(img, (5,5))
    img_Guassian = cv2.GaussianBlur(img,(5,5),0)
    img_median = cv2.medianBlur(img, 5)
    img_bilater = cv2.bilateralFilter(img,9,75,75)
    
    n = random.random()
    if n<=0.25:
        return img_mean,pts
    if n <= 0.5:
        return img_Guassian,pts
    if n <= 0.75:
        return img_median,pts
    if n <= 1:
        return img_bilater,pts

# 随机噪声
def randomNoise(img,pts):

    N = int(img.shape[0] * img.shape[1] * 0.001)
    for i in range(N): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255

    return img,pts

# 裁剪人脸下部
def randomCutBelowHalfFace(img,pts):

    height,width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    h = maxY - minY

    # enlarge to whole head
    minY -= 0.3 * h
    h = 1.3*h

    nose = pts[12,1]

    delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.2))

    ny2 = min(nose + delta_y2,height)
    
    if len(img.shape) == 3:
        img = img[: int(ny2),:, :]
    if len(img.shape) == 2:
        img = img[: int(ny2),:]

    return img,pts

# 裁剪人脸上部
def randomCutAboveHalfFace(img,pts):

    height,width = img.shape[0:2]

    minX,minY = pts.min(axis=0)
    maxX,maxY = pts.max(axis=0)

    h = maxY - minY

    # enlarge to whole head
    minY -= 0.3 * h
    h = 1.3*h

    nose = pts[12,1]

    delta_y2 = np.random.randint(int(h * 0.0), int(h * 0.5))

    ny1 = max(nose - delta_y2,0)
    
    if len(img.shape) == 3:
        img = img[int(ny1):,:, :]
    if len(img.shape) == 2:
        img = img[int(ny1):,:]

    pts[:,1] -= int(ny1)

    return img,pts

# 降低对比度，模拟逆光
def constrastImg(img):

    a = random.uniform(0.5,0.8)
    b = random.randint(50,100)
    contrast = -random.randint(50,100)

    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows,cols = img.shape
    for i in range(rows):
        for j in range(cols):

            img[i,j] = img[i,j]*a+b
            img[i,j] = img[i,j] * (contrast/127 + 1) - contrast

            if img[i,j]>255:           # 防止像素值越界（0~255）
                img[i,j]=255
            elif img[i,j]<0:           # 防止像素值越界（0~255）
                img[i,j]=0
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# 使用transform进行颜色抖动
def colorJitting(img):

    ColorJit = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
    # transform = transforms.Compose([transforms.ToPILImage(),colorJitting])
    # print('1',type(img))
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
    # print('2',type(img))
    img = ColorJit(img)
    # print('3',type(img))
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)

    return img

# 随机擦除
def randomErasing(img):

    RandomErasing_OB = RandomErasing(probability=0.5, sl=0.02, sh=0.1,device='cpu')
    # transform = transforms.Compose([RandomErasing,])

    # print('0',img.shape)
    img = img[np.newaxis]
    # print('a',img.shape)
    img = torch.from_numpy(img)
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
def randomAug_pts(img,pts,box=np.zeros((1,4)),target='eye'):

    if(1):

        # if 'eye' == target:
        #     img,pts = randomeResizeEye(img,pts)
        # elif 'mouth' == target:
        #     img,pts = randomeResizeMouth(img,pts)
        # else:
        #     sys.exit('target is wrong')
        # print('a',img.shape)
        img,pts = randomeResize(img,pts)
        # print('b',img.shape)
        # print(pts)
        # img = randomErasing(img)
        # print('c',img.shape)

        # if random.random() > 0.5:
        #     img,pts = randomBlur(img,pts)
        #     # print('d',img.shape)
        # if random.random() > 0.5:
        #     img = colorJitting(img)
            # print('e',img.shape)
        # if random.random() > 0.5:
        #     img,pts = randomFlip(img,pts)
            # print('f',img.shape)
        # if random.random() > 0.5:
        #     img,pts = randomeRotate(img,pts)
            # print('g',img.shape)

        # if ('eye' == target):
        #     if random.random() > 0.0:
        #         img,pts = randomCutBelowHalfFace(img,pts)
        # print('c',img.shape)
        # print(pts)
        # if ('mouth' == target):    
        #     if random.random() > 1.0:
        #         img,pts = randomCutAboveHalfFace(img,pts) 

    return img,pts

# Onet版本随机增强
def randomAug_Onet_pts(img,pts):

    img,pts = randomeOnetResize(img,pts)

    if random.random() > 0.5:
        img,pts = randomFlip(img,pts)

    return img,pts




if __name__ == '__main__':
    
    while True:

        imgFile = os.path.join('J:/img_Angle/','0000000000000000-170920-142651-144401-000006000180_1599.jpg')
        ptsFile = os.path.join('J:/img_Angle/','0000000000000000-170920-142651-144401-000006000180_1599.pts')

        img = cv2.imread(imgFile,-1)
        pts = getpoints(ptsFile)
        box = getbox(imgFile.replace('.jpg','.xml'))

        img,pts = randomAug_pts(img,pts,box,'mouth')

  
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(19):

            cv2.circle(img,(int(pts[i][0]),int(pts[i][1])),3,(255,0,0),-1)
            cv2.putText(img,str(i),(int(pts[i][0]),int(pts[i][1])), font, 0.4, (255, 255, 255), 1)

        cv2.imshow('img',img)
        cv2.waitKey(0)
