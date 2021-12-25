'''
@Author: Jiangtao
@Date: 2019-08-19 09:12:29
LastEditors: Jiangtao
LastEditTime: 2020-08-07 17:50:13
@Description: 检查bbox的可视化效果
'''
import cv2
import numpy as np  
import glob
import os
import xml.dom.minidom
import random
import math
from tqdm import tqdm
import time

def getlist(dir,extension,Random=False):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root,name))
                list[-1]  = list[-1].replace('\\','/')
    if Random:
        random.shuffle(list)
    return list

def randombaoguang(img):

    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    #将bgr转化为hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.astype(np.float)
    #获取v通道（颜色亮度通道），并做渐变性的增强
    img[:, :, 2] = np.where(img[:, :, 2] > 100, img[:, :, 2] + 20.0, img[:, :, 2])
    img[:, :, 2] = np.where(img[:, :, 2] > 150, img[:, :, 2] + 30.0, img[:, :, 2])
    img[:, :, 2] = np.where(img[:, :, 2] > 180, img[:, :, 2] + 40.0, img[:, :, 2])
    #令大于255的像素值等于255（防止溢出）
    img = np.where(img>255, 255, img)
    img = img.astype(np.uint8)
    res = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    return res
    
def functiona(img):

    x = random.choice([17,18,19])
    y = random.choice([2,3])

    img = x*math.log(1+img,y)
    return img

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


with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/detectLossList.txt','r') as f:
    lines = f.readlines()
annotations = lines


imgList = []
for annotation in annotations:

    strings = annotation.strip().split(' ')
    imgFile = ''

    for i in range(len(strings)-1):
        if i != 0:
            strings[i] = ' ' + strings[i]
        imgFile += strings[i]
    # imgFile = imgFile.replace('/jiangtao2/', 'J:/')
    imgList.append(imgFile)

# print(imgList[0:5])
# time.sleep(1000)

# DIR = 'D:/images_check/smoke/'

# imgList = getlist(DIR,'.jpg')
# imgList += getlist(DIR,'.JPG')


for img_file in tqdm(imgList[0:3500]):

    try:

    # if ('84659e86-9393-11eb-b3c5-d0c637e007cb' not in img_file):
    #     continue
    # print(img_file)
    img_file = img_file.replace('\\','/')
    img = cv2.imread(img_file,0)

    # if 1280 != img.shape[0]:
    #     continue

    xml_file = img_file
    xml_file = os.path.splitext(xml_file)[0] + '.xml'

    if not os.path.exists(xml_file):
        print('{} do not have xml'.format(img_file))
        continue

    boxes = getboxes(xml_file)
    # print(boxes)

    for box in boxes:

        minX = box[0]
        minY = box[1]
        maxX = box[2]
        maxY = box[3]


        # if(int(maxX) - int(minX) < 100) and ((1280 - int(maxX)) < 50):
        #     print(img_file)
        #     # os.remove(xml_file)
        #     # os.remove(img_file)
        #     continue

        cv2.rectangle(img,(int(float(minX)),int(float(minY))),(int(float(maxX)),int(float(maxY))),(255,0,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'leftup',(int(minX),int(minY)), font, 0.4, (0, 0, 255), 1)
        cv2.putText(img,'rightdown',(int(maxX),int(maxY)), font, 0.4, (0, 255, 0), 1)


    if(0):
        cv2.imshow('a',img)
        cv2.waitKey(0)
    if(1):
        targetDir = '/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/images/detectTest'
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        cv2.imwrite(os.path.join(targetDir,os.path.basename(img_file)),img)

