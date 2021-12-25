'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-12-24 18:41:25
@Description: 
'''

import inspect
import os
import random
import re
import sys
import time
import xml.dom.minidom

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import copy
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision import transforms
import os.path as osp
from copy import deepcopy
from tqdm import tqdm
import cv2
import os.path as osp
from ..transform import *

transformations = transforms.Compose([transforms.Resize(160),
transforms.RandomCrop(128), transforms.ToTensor()])
HARD_DATA = ['img_Angle','XR3.0','XR4.0']


def enlargeRoi(minx,miny,maxx,maxy,w,h,scale=0.00):

    roiw = maxx - minx
    roih = maxy - miny

    minx -= roiw*scale*2
    miny -= roih*scale

    maxx += roiw*scale*2
    maxy += roih*scale

    minx = max(0,minx)
    miny = max(0,miny)

    maxx = min(w,maxx)
    maxy = min(h,maxy)

    return [int(minx),int(miny),int(maxx),int(maxy)]

def fillterWrongDirect(imgPath):

    imgPath = imgPath.replace('\\','/')

    if('/eyeLeft' in imgPath):
        pass
    elif('/eyeRight' in imgPath):
        pass
    elif('/eyeMiddle/' in imgPath):
        pass
    elif('/eyeCenter/' in imgPath):
        pass
    elif('/eyeCentre/' in imgPath):
        pass
    elif('/up/' in imgPath):
        pass
    elif('Down' in imgPath):
        pass
    else:
        return 0

    return 1

def isHard(imgFile):
    for hardformat in HARD_DATA:
        if hardformat in imgFile:
            return True
    return False

def pts21to19(pts):

    assert pts.shape[0] == 21

    ptsNew = np.zeros((19,2))

    for i in range(5):
        ptsNew[i] = pts[i]
    ptsNew[5] = pts[6]

    for i in range(4):
        ptsNew[i+6] = pts[i+7]

    ptsNew[10] = pts[12]
    for i in range(8):
        ptsNew[i+11] = pts[i+13]

    return ptsNew

def pts19to21(pts):

    assert pts.shape[0] == 19

    ptsNew = np.zeros((21,2))

    for i in range(5):
        ptsNew[i] = pts[i]
    ptsNew[5] = pts[1:5,].mean(axis=0)
    ptsNew[6] = pts[5]

    for i in range(5):
        ptsNew[i+7] = pts[i+6]
    ptsNew[11] = pts[6:10,].mean(axis=0)
    ptsNew[12] = pts[10]

    for i in range(8):
        ptsNew[i+13] = pts[i+11]

    return ptsNew

def revisePts(pts,width,height):

    assert pts.shape[1] == 2
    nPoints = pts.shape[0]
    for i in range(nPoints):

        if pts[i][0] < 0:
            pts[i][0] = 0

        if pts[i][0] > width:
            pts[i][0] = width

        if pts[i][1] < 0:
            pts[i][1] = 0

        if pts[i][1] > height:
            pts[i][1] = height

    return pts

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def isLargeAngle(ptsFile):

    # pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1).reshape((-1,2))

    # if 21 == pts.shape[0]:
    #     left = pts[0,0]
    #     mid = pts[6,0]
    #     right = pts[12,0]

    # elif 19 == pts.shape[0]:
    #     left = pts[0,0]
    #     mid = pts[5,0]
    #     right = pts[10,0]

    # scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
    # scale2 = abs((right-mid) / ((mid-left)+1e-5))

    if 'right' in ptsFile:
        return 1
    elif 'left' in ptsFile:
        return -1
    else:
        return 0

def ifAccord(ptsFile,target):

    if 'eye' == target:
        return 1
    
    if 'mouth' == target:
        if'unmask' in ptsFile:
            return 1
        elif 'newFormat' in ptsFile:
            with open(ptsFile,'r') as f:
                annotations = f.readlines()
                if 'npoints: 21' == annotations[1].strip():
                    return 1
                else:
                    return 0
        else:
            return 0

def mastOnImage(img,x1,y1,x2,y2):

    assert len(img.shape) == 2,'img dims is wrong'
    assert type(x1) is int and type(x1) is int,'location is not int'

    resized_im = np.random.randint(255,size=(y2-y1,x2-x1))
    img[y1:y2,x1:x2] = resized_im
    return img

def DebugPrint(value):

    print(type(value))
    print(value)
    
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

def getPts(ptsFile):

    with open(ptsFile,'r') as f:
        lines = f.readlines()


    assert int(len(lines) - 1) % 24 == 0

    nFace = int(len(lines) - 1) / 24

    res = []
    for i in range(int(nFace)):
        begin = 24*i + 3
        end = 24*i + 24

        pts = lines[begin:end]
        pts = [a.strip().split(' ') for a in pts]
        newpts = []
        for singlepoint in pts:
            singleres = []
            for axis in singlepoint:
                axis = int(float(axis))
                singleres.append(axis)
            newpts.append(singleres)

        pts = np.array(pts)
        newpts = np.array(newpts)
        # print(pts)
        res.append(newpts)
    return res

def getnpoints(ptsFile):

    with open(ptsFile,'r') as f:
        lines = f.readlines()

    assert int(len(lines) - 1) % 24 == 0

    nFace = int(len(lines) - 1) / 24
    nFace = int(nFace)

    res = []
    for i in range(nFace):
        npoints = lines[24*i+1].strip().split(' ')[-1]
        npoints = int(npoints)
        res.append(npoints)
    return res

def isValidXml(xmlFile):

    # 检查图片是否能打开
    filename = os.path.splitext(xmlFile)[0]
    imgFile = filename + '.jpg'
    try:
        img = cv2.imread(imgFile,0)
        height, width = img.shape[0:2]
    except:
        print('{} can not open'.format(imgFile))
        return False

    # 检查框的质量，特别小的直接过滤,超过图像宽高的也过滤
    boxes = getboxes(xmlFile)

    for box in boxes:
        minX,minY = box[0],box[1]
        maxX,maxY = box[2],box[3]

        if (maxX-minX) <= 50 or (maxY-minY) <= 50:
            print('{} have boxes is small than 50'.format(imgFile))
            return False

        if(maxX > width or maxY > height or minX < 0 or minY < 0):
            print('{} have boxes larger than img'.format(imgFile))
            return False

    return True

def isValidPts(ptsFile):

    # 检查图片是否能打开
    filename = os.path.splitext(ptsFile)[0]
    imgFile = filename + '.jpg'
    try:
        img = cv2.imread(imgFile,0)
        height, width = img.shape[0:2]
    except:
        # print('{} can not open'.format(imgFile))
        return False

    # 检查特征点的质量，超过图像宽高的过滤
    try:
        ptsList = getPts(ptsFile)
        npointsList = getnpoints(ptsFile)
    except:
        print('{} have wrong'.format(ptsFile))
        return False

    assert len(ptsList) == len(npointsList), "error in ptsFile"

    for pts in ptsList:

        minX,minY = pts.min(axis=0)
        maxX,maxY = pts.max(axis=0)
        w = maxX - minX
        h = maxY - minY

        # if(pts[14][1] <= (pts[6][1] + 50)):
        #     return False

        # if(pts[13][1] <= (pts[14][1] + 50)):
        #     return False

        # if(maxX > width+10 or maxY > height+10 or minX < -10 or minY < -10 or w < 50 or h < 50):
        #     # print('{} have pts larger than img'.format(imgFile))
        #     return False

        if(pts[1][1] == maxY or pts[2][0] == maxX):
            return False

    return True

def isValidImg(imgFile):

    if('aflw' in imgFile):
        return False

    try:
        img = cv2.imread(imgFile,0)
        height, width = img.shape[0:2]
    except:
        return False

    filename = os.path.splitext(imgFile)[0]
    ptsFile = filename + '.pts'

    # 检查特征点的质量，超过图像宽高的过滤
    ptsList = getPts(ptsFile)
    npointsList = getnpoints(ptsFile)

    if(len(ptsList) != len(npointsList)):
        return False

    for pts in ptsList:

        minX,minY = pts.min(axis=0)
        maxX,maxY = pts.max(axis=0)
        w = maxX - minX
        h = maxY - minY

        if(maxX > width+10 or maxY > height+10 or minX < -10 or minY < -10 or w < 50 or h < 50):
            return False
    
    return True

def easyCheckPts(imgFile):

    filename = os.path.splitext(imgFile)[0]
    ptsFile = filename + '.pts'

    if not os.path.exists(ptsFile):
        return False

    try:
        pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1)
        ptsEye = pts[0:11,]
    except:
        return False

    minX,minY = ptsEye.min(axis=0)
    maxX,maxY = ptsEye.max(axis=0)

    w = maxX - minX
    h = maxY - minY

    if(w < 100 or h < 15):
        return False

    img = cv2.imread(imgFile,0)
    if img is None:
        return False

    return True

def easyGetPts(ptsFile):

    pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1)
    return pts

def getpoints(ptsFile):

    pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1).reshape((-1,2))

    if 21 == pts.shape[0]:
        pass

    elif 19 == pts.shape[0]:
        pass
    else:
        print('some wrong in pts')
        sys.exit()

    return pts

def getlist(dir,extension):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            # print(os.path.join(root, name))
            pass
        for name in files:
            filename,ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root,name))
    return list

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines
    
def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d
    
def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params
    
def get_eye_pts_from_68_11(line):

    if (len(line) != 6) and (len(line) != 138):
        print('len(line):',len(line))
        print(line)
    # print('len(line):',len(line))
    # print(line)
    if len(line) == 6:
        eyePts = np.zeros((22,))
    else:
        pts = line[2:]
        pts = np.array(pts).reshape((68,1,2))
        eyePts = np.concatenate((pts[0,:],pts[36,:],pts[37,:],pts[39,:],
                                pts[41,:],pts[27,:],pts[42,:],pts[44,:],
                                pts[45,:],pts[46,:],pts[16,:]),axis=0)
    
    eyePts = eyePts.reshape((22,))

    return eyePts

def get_mouth_pts_from_68_8(line):
    
    if len(line) == 6:
        mouthPts = np.zeros((16,))

    else:
        pts = line[2:]
        pts = np.array(pts).reshape((68,1,2))
        mouthPts = np.concatenate((pts[8,:],pts[30,:],pts[48,:],pts[51,:],
                                    pts[54,:],pts[57,:],pts[62,:],pts[66,:],),axis=0)

    mouthPts = mouthPts.reshape((16,))

    return mouthPts

def read_image_list_with_multi3(path):

    with open(path, "r") as file:

        imgPath = []
        eyePoints = []
        mouthPoints = []
        face = []
        detect = []

        for line in file.readlines():

            line = line.strip().split(' ')

            curLabelEye = get_eye_pts_from_68_11(line)
            curLabelMouth = get_mouth_pts_from_68_8(line)

            imgFile = '/mnt/workspace/jiangtao/' + line[0]

            if not os.path.exists(imgFile):
                # print('do not have')
                # print(imgFile)
                # time.sleep(10)
                continue
            

            imgPath.append(line[0])
            eyePoints.append(curLabelEye)
            mouthPoints.append(curLabelMouth)
            face.append(int(line[1]))
            detect.append(np.array([line[2],line[3],line[4],line[5]]))


    return imgPath, eyePoints, mouthPoints, face, detect

def image_sample_with_multi3(pathList):

    imagepathList = []
    eyeList = []
    mouthList = []
    labelList = []
    detectList = []

    shuffleList = []

    newImagepathList = []
    newLabelList = []
    newEyeList = []
    newMouthList = []
    newDetectList = []

    for i in range(len(pathList)):
        
        curImagepathList, curEyeList, curMouthList, curLabelList, curDetectList = read_image_list_with_multi3(pathList[i])

        imagepathList.extend(curImagepathList)
        eyeList.extend(curEyeList)
        mouthList.extend(curMouthList)
        labelList.extend(curLabelList)
        detectList.extend(curDetectList)

    if len(imagepathList) != len(labelList) or len(mouthList) != len(eyeList):
        print(len(imagepathList))
        print(len(labelList))
        print("image_smaple has some problem!\n")
        sys.exit()

    for i in range(len(imagepathList)):
        curList = []

        curList.append(imagepathList[i])
        curList.append(eyeList[i])
        curList.append(mouthList[i])
        curList.append(labelList[i])
        curList.append(detectList[i])

        shuffleList.append(curList)

    random.seed(0)
    random.shuffle(shuffleList)

    for i in range(len(shuffleList)):

        newImagepathList.append(shuffleList[i][0])
        newEyeList.append(shuffleList[i][1])
        newMouthList.append(shuffleList[i][2])
        newLabelList.append(shuffleList[i][3])
        newDetectList.append(shuffleList[i][4])

    return newImagepathList, newEyeList, newMouthList, newLabelList, newDetectList

def checkRight(imgPathList,labelPathList):

    print(imgPathList[0:5])
    print(labelPathList[0:5])

    newimgPathList = []
    newlabelPathList = []
    for i in tqdm(list(range(len(imgPathList)))):

        imgFile = imgPathList[i]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        try:
            ptsList = getPts(ptsFile)
            pts = ptsList[labelPathList[i]]
        except:
            print('{} have wrong'.format(imgPathList[i]))
            print(ptsList)
            print(labelPathList[i])
            imgPathList.remove(imgPathList[i])
            labelPathList.remove(labelPathList[i])
            continue

        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)
        left = pts[0][0]
        mid = pts[5][0]
        right = pts[10][0]

        scale = abs(mid-left) / (abs(right - mid) +0.001)

        if(scale > 10):
            newimgPathList.append(imgPathList[i])
            newlabelPathList.append(labelPathList[i])
    
    return newimgPathList,newlabelPathList

def checkLeft(imgPathList,labelPathList):

    print(imgPathList[0:5])
    print(labelPathList[0:5])

    newimgPathList = []
    newlabelPathList = []
    for i in tqdm(list(range(len(imgPathList)))):

        imgFile = imgPathList[i]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        try:
            ptsList = getPts(ptsFile)
            pts = ptsList[labelPathList[i]]
        except:
            print('{} have wrong'.format(imgPathList[i]))
            print(ptsList)
            print(labelPathList[i])
            imgPathList.remove(imgPathList[i])
            labelPathList.remove(labelPathList[i])
            continue

        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)
        left = pts[0][0]
        mid = pts[5][0]
        right = pts[10][0]

        scale = abs(right - mid) / (abs(mid - left) +0.001)

        if(scale > 10):
            newimgPathList.append(imgPathList[i])
            newlabelPathList.append(labelPathList[i])

    return newimgPathList,newlabelPathList

def filterWrinkleImg(imgFile):

    dirname = osp.dirname(imgFile)
    dirname = osp.basename(dirname)

    try:
        label = int(dirname)
    except:
        return False

    if label > 2 or label < 0:
        os.remove(imgFile)
        return False
    
    return True

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

class DatasetWithAngleMulti3(torchDataset):

    def __init__(self, imglistPath, inputSize, img_tf, label_tf, imgChannel=1,isTrain='train'):

        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList, self.eyeList, self.mouthList, self.labelList, self.detectList = image_sample_with_multi3(imglistPath)
        else:
            self.imgPathList, self.eyeList, self.mouthList, self.labelList, self.detectList = read_image_list_with_multi3(imglistPath)

        if isTrain == 'train':

            nTrain = int(len(self.imgPathList) * 0.9)
            self.imgPathList = self.imgPathList[0:nTrain]
            self.eyeList = self.eyeList[0:nTrain]
            self.mouthList = self.mouthList[0:nTrain]
            self.labelList = self.labelList[0:nTrain]
            self.detectList = self.detectList[0:nTrain]

            print(len(self.imgPathList))
            print(len(self.eyeList))
            print(len(self.mouthList))
            print(len(self.labelList))
            print(len(self.detectList))

        if isTrain == 'val':

            nTrain = int(len(self.imgPathList) * 0.9)

            self.imgPathList = self.imgPathList[nTrain:]
            self.eyeList = self.eyeList[nTrain:]
            self.mouthList = self.mouthList[nTrain:]
            self.labelList = self.labelList[nTrain:]
            self.detectList = self.detectList[nTrain:]

        if isTrain == 'trainval':

            self.imgPathList = self.imgPathList
            self.eyeList = self.eyeList
            self.mouthList = self.mouthList
            self.labelList = self.labelList
            self.detectList = self.detectList
        
        self.img_tf = img_tf
        self.label_tf = label_tf
        self.num = 0
        self.channel = imgChannel

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        if sys.platform == "win32":
            if(self.channel == 1):
                img = cv2.imread("C:/code/utilisation/" + self.imgPathList[index], 0)
            else:
                img = cv2.imread("C:/code/utilisation/" + self.imgPathList[index], 1)

        elif sys.platform == "linux":
            if(self.channel == 1):
                img = cv2.imread("/mnt/workspace/jiangtao/" + self.imgPathList[index].replace('\\','/'), 0)
            else:
                img = cv2.imread("/mnt/workspace/jiangtao/" + self.imgPathList[index].replace('\\','/'), 1)


        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and is {}\n".format(self.imgPathList[index]))
            print("/mnt/workspace/jiangtao/128_new" + self.imgPathList[index].replace('\\','/'))
            sys.exit()
            

        if self.img_tf is not None:
            img = self.img_tf(img)

        return img, self.eyeList[index], self.mouthList[index], self.labelList[index], self.detectList[index]

    def collate_fn(self, batch):
        
        images = list()
        eye = list()
        mouth = list()
        label = list()
        detect = list()

        for b in batch:

            images.append(b[0].float())
            eye.append(b[1].tolist())
            mouth.append(b[2].tolist())
            label.append(b[3])
            detect.append(b[4].tolist())
            
        eye = np.array(eye,dtype=np.float64)
        mouth = np.array(mouth,dtype=np.float64)
        detect = np.array(detect,dtype=np.float64)

        images = torch.stack(images, dim=0)
        images = torch.FloatTensor(images)
        eye = torch.FloatTensor(eye)
        mouth = torch.FloatTensor(mouth)
        label = torch.FloatTensor(label)
        detect = torch.FloatTensor(detect)

        return images, eye, mouth, label, detect

class Pose_300W_LP(torchDataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform=transformations, img_ext='.jpg', annot_ext='.mat', image_mode='L',isTrain='train'):

        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        
        filename_list = get_list_from_filenames(filename_path)
        nTrain = int(len(filename_list) * 0.8)

        if isTrain == 'train':
            filename_list = filename_list[0:nTrain]
        if isTrain == 'val':
            filename_list = filename_list[nTrain:]

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw                          
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels

    def __len__(self):
        # 122,450
        return self.length

class DatasetDetect(torchDataset):

    def __init__(self, imgDir, size, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        # self.fileList = []
        # for i in range(len(imgDir)):
        #     self.fileList += getlist(imgDir[i],'.jpg')

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))
        
        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        from tqdm import tqdm
        for imgPath in tqdm(self.fileList):
            filename = os.path.splitext(imgPath)[0]
            xmlPath = filename + '.xml'
            if os.path.exists(xmlPath):
                if isValidXml(xmlPath):
                    self.imgPathList.append(imgPath)
                    self.labelPathList.append(xmlPath)
                else:
                    with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/errorSamples.txt','a') as f:
                        f.write(imgPath)
                        f.write('\n')


        assert len(self.imgPathList) == len(self.labelPathList)


        # # split train and validation
        # nTrain = int(len(self.imgPathList) * 0.9)

        # if isTrain == 'train':
        #     self.imgPathList = self.imgPathList[0:nTrain]
        #     self.labelPathList = self.labelPathList[0:nTrain]

        # if isTrain == 'val':
        #     self.imgPathList = self.imgPathList[nTrain:]
        #     self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)

        # xml中仅包含一个人脸
        # box = getbox(self.labelPathList[index])

        # xml中包含多个人脸
        boxes = getboxes(self.labelPathList[index])
        boxes_cp = deepcopy(boxes)
        indexBox = np.random.randint(boxes.shape[0],size=1)[0]
        box = boxes[indexBox].reshape((1,4))

        try:
            img_new, box_new = randomAug_box(img,box)
            height, width = img_new.shape[0:2]
        except:
            print(self.imgPathList[index])
            print(img.shape)
            print(box)
            sys.exit('something went wrong')


        x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
        w = x2 - x1
        h = y2 - y1

        if(w<=2 or h<=2):
            print(self.imgPathList[index])
            sys.exit('wrong in data')


        dice = random.random()

        # 保持合适的人脸比例，人头一般比较长
        # if dice < 0.5:
        if dice < 0.25:
            delta_x1 = np.random.randint(0,int(w * 0.25))
            delta_y1 = np.random.randint(0,int(h * 0.25))
            delta_x2 = np.random.randint(0,int(w * 0.25)) 
            delta_y2 = np.random.randint(0,int(h * 0.25)) 

        # elif (dice>=0.5) and (dice<0.95):
        elif (dice>=0.25) and (dice<0.5):
            delta_x1 = np.random.randint(int(w * 0.25), int(w * 0.50))
            delta_y1 = np.random.randint(int(h * 0.25), int(h * 0.50))
            delta_x2 = np.random.randint(int(w * 0.25), int(w * 0.50)) 
            delta_y2 = np.random.randint(int(h * 0.25), int(h * 0.50)) 

        elif (dice>=0.50) and (dice<0.75):
            delta_x1 = np.random.randint(int(w * 0.50), int(w * 1.0))
            delta_y1 = np.random.randint(int(h * 0.25), int(h * 0.50))
            delta_x2 = np.random.randint(int(w * 0.50), int(w * 1.0)) 
            delta_y2 = np.random.randint(int(h * 0.25), int(h * 0.50)) 

        #残缺人脸追踪设置,上下左右的残缺系数都是0.5
        # elif dice >= 0.95:
        elif dice >= 0.75:

            deformity = random.random()

            if deformity<0.25:
                delta_x1 = -np.random.randint(0,int(w * 0.3))
                delta_y1 = np.random.randint(0,int(h * 0.3))
                delta_x2 = np.random.randint(0,int(w * 0.3)) 
                delta_y2 = np.random.randint(0,int(h * 0.3)) 
            if deformity>=0.25 and deformity<0.5:
                delta_x1 = np.random.randint(0,int(w * 0.3))
                delta_y1 = np.random.randint(0,int(h * 0.3))
                delta_x2 = -np.random.randint(0,int(w * 0.3)) 
                delta_y2 = np.random.randint(0,int(h * 0.3)) 
            if deformity>=0.5 and deformity<0.75:
                delta_x1 = np.random.randint(0,int(w * 0.3))
                delta_y1 = -np.random.randint(0,int(h * 0.3))
                delta_x2 = np.random.randint(0,int(w * 0.3)) 
                delta_y2 = np.random.randint(0,int(h * 0.3)) 
            if deformity>=0.75:
                delta_x1 = np.random.randint(0,int(w * 0.3))
                delta_y1 = np.random.randint(0,int(h * 0.3))
                delta_x2 = np.random.randint(0,int(w * 0.3)) 
                delta_y2 = -np.random.randint(0,int(h * 0.3)) 

        nx1 = max(x1 - delta_x1,0)
        ny1 = max(y1 - delta_y1,0)
        nx2 = min(x2 + delta_x2,width)
        ny2 = min(y2 + delta_y2,height)

        # 将点归一化到裁剪区域中
        x1 = (x1 - nx1) * 128 / (nx2 - nx1)
        y1 = (y1 - ny1) * 128 / (ny2 - ny1)
    
        x1 = x1 / 128.0000000000
        y1 = y1 / 128.0000000000

        x2 = (x2 - nx1) * 128 / (nx2 - nx1)
        y2 = (y2 - ny1) * 128 / (ny2 - ny1)
    
        x2 = x2 / 128.0000000000
        y2 = y2 / 128.0000000000

        # print('[{},{},{},{}]'.format(nx1,ny1,nx2,ny2))
        # print('boxes:',boxes)
        # print('boxes_cp:',boxes_cp)
        # print('box:',box)
        # print('bbbbbbbbbbbbbbbbbbbbbbbb')
        # time.sleep(10000)

        try:
            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        except:
            print(self.imgPathList[index])
            print(boxes_cp)
            print(boxes)
            print(box)
            print(img_new.shape)
            print(box_new)
            print('{} {} {} {}'.format(delta_x1,delta_y1,delta_x2,delta_y2))
            print('{} {} {} {}'.format(ny1,ny2,nx1,nx2))
            print(dice)

            sys.exit('wrong in data')

        resized_im = resized_im[np.newaxis,]

        return torch.FloatTensor(resized_im),torch.FloatTensor([x1, y1, x2, y2]).clamp_(min=0,max=1)

class DatasetAlignmentNew(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):
        
        self.size = size
        self.channel = channel
        self.target = target

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')

        fileListApp = []
        for imgFile in self.fileList:
            if ('img_Angle' in imgFile):
                for _ in range(20):
                    fileListApp.append(imgFile)
        self.fileList += fileListApp

        self.fileList.sort()
        random.seed(612)
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            ptsPath = imgPath.replace('.jpg','.pts')
            if os.path.exists(ptsPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(ptsPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        img = cv2.imread(self.imgPathList[index],1)
        pts = getpoints(self.labelPathList[index])
        if(19 == pts.shape[0]):
            pts = pts19to21(pts)
        box = getbox(self.imgPathList[index].replace('.jpg','.xml'))

        # randomaug img and pts
        img_ori, pts_ori = randomAug_pts(img,pts,box,self.target)
        assert img_ori.shape[2] == img.shape[2]

        # process pts
        height, width = img_ori.shape[0:2]
        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('some wrong in dataset')

        # process img,set channel,and totensor
        if 3 == self.channel:
            resized_im = resized_im.astype('float') * 0.0039216
        if 1 == self.channel:
            resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
            resized_im = resized_im[:,:,np.newaxis]

        resized_im = resized_im.transpose(2,0,1)
        
        eyePts = pts_ori[0:13,:].reshape((-1))
        mouthPts= pts_ori[13:,:].reshape((-1))
        
        return torch.FloatTensor(resized_im), torch.FloatTensor([eyePts]).clamp_(min=0,max=1), torch.FloatTensor([mouthPts]).clamp_(min=0,max=1)

class DatasetAlignmentOld(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 记录包含21点的图片名，和对应的人头序号。
        self.imgPathList21 = []
        self.labelPathList21 = []

        # 记录包含13点的图片名，和对应的人头序号。
        self.imgPathList13 = []
        self.labelPathList13 = []

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # # 检查是否包含错误样本
        if(0):
            for i,imgPath in tqdm(enumerate(self.fileList)):

                filename = os.path.splitext(imgPath)[0]
                ptsPath = filename + '.pts'

                if os.path.exists(ptsPath):
                    if isValidPts(ptsPath):
                        continue
                    else:
                        self.fileList.pop(i)
                        with open('./errorSamplesAlign.txt','a') as f:
                            f.write(imgPath)
                            f.write('\n')
                else:
                    self.fileList.pop(i)
                    with open('./errorSamplesAlign.txt','a') as f:
                        f.write(imgPath)
                        f.write('\n')
        print('After check have {} imgs'.format(len(self.fileList)))


        # split to 21 and 13 List, and check
        for imgPath in tqdm(self.fileList):
            filename = os.path.splitext(imgPath)[0]
            ptsPath = filename + '.pts'
            if os.path.exists(ptsPath):

                ptsList = getPts(ptsPath)
                npointsList = getnpoints(ptsPath)

                assert len(ptsList) == len(npointsList), 'something is wrong'

                for j,npoints in enumerate(npointsList):

                    if 21 == npoints:
                        self.imgPathList21.append(imgPath)
                        self.labelPathList21.append(j)
                    elif 13 == npoints:
                        self.imgPathList13.append(imgPath)
                        self.labelPathList13.append(j)
                    else:
                        sys.exit('{} have wrong num of points'.format(imgPath))
                # except:
                #     continue

        assert len(self.imgPathList21) == len(self.labelPathList21)
        assert len(self.imgPathList13) == len(self.labelPathList13)

        print('len(self.imgPathList21):',len(self.imgPathList21))
        print('len(self.imgPathList13):',len(self.imgPathList13))

        if 'eye' == self.target:
            self.imgPathList = self.imgPathList21 + self.imgPathList13
            self.labelPathList = self.labelPathList21 + self.labelPathList13
        if 'mouth' == self.target:
            self.imgPathList = self.imgPathList21
            self.labelPathList = self.labelPathList21

        print('len(self.imgPathList):',len(self.imgPathList))
        # time.sleep(1000)
        # self.imgPathList,self.labelPathList = checkRight(self.imgPathList,self.labelPathList)
        # self.imgPathList,self.labelPathList = checkLeft(self.imgPathList,self.labelPathList)

        print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[self.labelPathList[index]]


        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        # randomaug img and pts
        # try:
        #     img_ori, pts_ori = randomAug_pts(img,pts)
        # except:
        #     print(img.shape)
        #     print(pts)
        #     sys.exit('randomAug_pts:{} have wrong'.format(imgFile))

        xmlFile = imgFile.replace('.jpg','.xml')
        if osp.exists(xmlFile):
            box = getboxes(xmlFile)[0]
            box = enlargeRoi(box[0],box[1],box[2],box[3],width,height,0.15)
            minx,miny,maxx,maxy = box
            img_ori = img[miny:maxy,minx:maxx,]
            pts_ori = pts - [minx,miny]
        else:
            img_ori, pts_ori = randomAug_pts(img,pts)

        # process pts
        try:
            height, width = img_ori.shape[0:2]
        except:
            print(self.imgPathList[index])
            sys.exit('cannot get height,width')

        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('can not resize')

        # process img,set channel,and totensor
        # if 3 == self.channel:
        #     resized_im = resized_im.astype('float') * 0.0039216
        # if 1 == self.channel:
        #     resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
        #     resized_im = resized_im[:,:,np.newaxis]
        # print(resized_im[0][0])
        resized_im = resized_im * 0.0039216
        # print(resized_im[0][0])
        # time.sleep(1000)
        resized_im = resized_im[:,:,np.newaxis]

        # print(resized_im.shape)
        # print(resized_im[0][0])
        # time.sleep(1000)

        resized_im = resized_im.transpose(2,0,1)
        # test = torch.FloatTensor(resized_im)
        # print(resized_im.shape)
        # print(test.shape)
        # print(resized_im[0][0][0])
        # print(resized_im[0][0][1])
        # print(test[0][0][0])
        # print(test[0][0][1])

        eyePts = pts_ori[0:11,:].reshape((-1))
        mouthPts= pts_ori[11:,:].reshape((-1))
        # print(eyePts)
        # print(torch.FloatTensor([eyePts]).clamp_(min=0,max=1).shape)
        # time.sleep(1000)
        # print('a',type(resized_im))

        return resized_im, torch.FloatTensor([eyePts]), torch.FloatTensor([mouthPts])

class DatasetAlignmentOldRight(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 记录包含21点的图片名，和对应的人头序号。
        self.imgPathList21 = []
        self.labelPathList21 = []

        # 记录包含13点的图片名，和对应的人头序号。
        self.imgPathList13 = []
        self.labelPathList13 = []

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # # 检查是否包含错误样本
        if(0):
            for i,imgPath in tqdm(enumerate(self.fileList)):

                filename = os.path.splitext(imgPath)[0]
                ptsPath = filename + '.pts'

                if os.path.exists(ptsPath):
                    if isValidPts(ptsPath):
                        continue
                    else:
                        self.fileList.pop(i)
                        with open('./errorSamplesAlign.txt','a') as f:
                            f.write(imgPath)
                            f.write('\n')
                else:
                    self.fileList.pop(i)
                    with open('./errorSamplesAlign.txt','a') as f:
                        f.write(imgPath)
                        f.write('\n')
        print('After check have {} imgs'.format(len(self.fileList)))


        # split to 21 and 13 List, and check
        for imgPath in tqdm(self.fileList):
            filename = os.path.splitext(imgPath)[0]
            ptsPath = filename + '.pts'
            if os.path.exists(ptsPath):

                ptsList = getPts(ptsPath)
                npointsList = getnpoints(ptsPath)

                assert len(ptsList) == len(npointsList), 'something is wrong'

                for j,npoints in enumerate(npointsList):

                    if 21 == npoints:
                        self.imgPathList21.append(imgPath)
                        self.labelPathList21.append(j)
                    elif 13 == npoints:
                        self.imgPathList13.append(imgPath)
                        self.labelPathList13.append(j)
                    else:
                        sys.exit('{} have wrong num of points'.format(imgPath))
                # except:
                #     continue

        assert len(self.imgPathList21) == len(self.labelPathList21)
        assert len(self.imgPathList13) == len(self.labelPathList13)

        print('len(self.imgPathList21):',len(self.imgPathList21))
        print('len(self.imgPathList13):',len(self.imgPathList13))

        if 'eye' == self.target:
            self.imgPathList = self.imgPathList21 + self.imgPathList13
            self.labelPathList = self.labelPathList21 + self.labelPathList13
        if 'mouth' == self.target:
            self.imgPathList = self.imgPathList21
            self.labelPathList = self.labelPathList21

        print('len(self.imgPathList):',len(self.imgPathList))

        self.imgPathList,self.labelPathList = checkRight(self.imgPathList,self.labelPathList)
        # self.imgPathList,self.labelPathList = checkLeft(self.imgPathList,self.labelPathList)

        print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[self.labelPathList[index]]


        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        # randomaug img and pts
        # try:
        #     img_ori, pts_ori = randomAug_pts(img,pts)
        # except:
        #     print(img.shape)
        #     print(pts)
        #     sys.exit('randomAug_pts:{} have wrong'.format(imgFile))

        xmlFile = imgFile.replace('.jpg','.xml')
        if osp.exists(xmlFile):
            box = getboxes(xmlFile)[0]
            box = enlargeRoi(box[0],box[1],box[2],box[3],width,height,0.15)
            minx,miny,maxx,maxy = box
            img_ori = img[miny:maxy,minx:maxx,]
            pts_ori = pts - [minx,miny]
        else:
            img_ori, pts_ori = randomAug_pts(img,pts)

        # process pts
        try:
            height, width = img_ori.shape[0:2]
        except:
            print(self.imgPathList[index])
            sys.exit('cannot get height,width')

        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('can not resize')

        # process img,set channel,and totensor
        # if 3 == self.channel:
        #     resized_im = resized_im.astype('float') * 0.0039216
        # if 1 == self.channel:
        #     resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
        #     resized_im = resized_im[:,:,np.newaxis]
        # print(resized_im[0][0])
        resized_im = resized_im * 0.0039216
        # print(resized_im[0][0])
        # time.sleep(1000)
        resized_im = resized_im[:,:,np.newaxis]

        # print(resized_im.shape)
        # print(resized_im[0][0])
        # time.sleep(1000)

        resized_im = resized_im.transpose(2,0,1)
        # test = torch.FloatTensor(resized_im)
        # print(resized_im.shape)
        # print(test.shape)
        # print(resized_im[0][0][0])
        # print(resized_im[0][0][1])
        # print(test[0][0][0])
        # print(test[0][0][1])

        eyePts = pts_ori[0:11,:].reshape((-1))
        mouthPts= pts_ori[11:,:].reshape((-1))
        # print(eyePts)
        # print(torch.FloatTensor([eyePts]).clamp_(min=0,max=1).shape)
        # time.sleep(1000)
        # print('a',type(resized_im))

        return resized_im, torch.FloatTensor([eyePts]), torch.FloatTensor([mouthPts])

class DatasetAlignmentOldLeft(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 记录包含21点的图片名，和对应的人头序号。
        self.imgPathList21 = []
        self.labelPathList21 = []

        # 记录包含13点的图片名，和对应的人头序号。
        self.imgPathList13 = []
        self.labelPathList13 = []

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # # 检查是否包含错误样本
        if(0):
            for i,imgPath in tqdm(enumerate(self.fileList)):

                filename = os.path.splitext(imgPath)[0]
                ptsPath = filename + '.pts'

                if os.path.exists(ptsPath):
                    if isValidPts(ptsPath):
                        continue
                    else:
                        self.fileList.pop(i)
                        with open('./errorSamplesAlign.txt','a') as f:
                            f.write(imgPath)
                            f.write('\n')
                else:
                    self.fileList.pop(i)
                    with open('./errorSamplesAlign.txt','a') as f:
                        f.write(imgPath)
                        f.write('\n')
        print('After check have {} imgs'.format(len(self.fileList)))


        # split to 21 and 13 List, and check
        for imgPath in tqdm(self.fileList):
            filename = os.path.splitext(imgPath)[0]
            ptsPath = filename + '.pts'
            if os.path.exists(ptsPath):

                ptsList = getPts(ptsPath)
                npointsList = getnpoints(ptsPath)

                assert len(ptsList) == len(npointsList), 'something is wrong'

                for j,npoints in enumerate(npointsList):

                    if 21 == npoints:
                        self.imgPathList21.append(imgPath)
                        self.labelPathList21.append(j)
                    elif 13 == npoints:
                        self.imgPathList13.append(imgPath)
                        self.labelPathList13.append(j)
                    else:
                        sys.exit('{} have wrong num of points'.format(imgPath))
                # except:
                #     continue

        assert len(self.imgPathList21) == len(self.labelPathList21)
        assert len(self.imgPathList13) == len(self.labelPathList13)

        print('len(self.imgPathList21):',len(self.imgPathList21))
        print('len(self.imgPathList13):',len(self.imgPathList13))

        if 'eye' == self.target:
            self.imgPathList = self.imgPathList21 + self.imgPathList13
            self.labelPathList = self.labelPathList21 + self.labelPathList13
        if 'mouth' == self.target:
            self.imgPathList = self.imgPathList21
            self.labelPathList = self.labelPathList21

        print('len(self.imgPathList):',len(self.imgPathList))

        # self.imgPathList,self.labelPathList = checkRight(self.imgPathList,self.labelPathList)
        self.imgPathList,self.labelPathList = checkLeft(self.imgPathList,self.labelPathList)

        print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[self.labelPathList[index]]


        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        # randomaug img and pts
        # try:
        #     img_ori, pts_ori = randomAug_pts(img,pts)
        # except:
        #     print(img.shape)
        #     print(pts)
        #     sys.exit('randomAug_pts:{} have wrong'.format(imgFile))

        xmlFile = imgFile.replace('.jpg','.xml')
        if osp.exists(xmlFile):
            box = getboxes(xmlFile)[0]
            box = enlargeRoi(box[0],box[1],box[2],box[3],width,height,0.15)
            minx,miny,maxx,maxy = box
            img_ori = img[miny:maxy,minx:maxx,]
            pts_ori = pts - [minx,miny]
        else:
            img_ori, pts_ori = randomAug_pts(img,pts)

        # process pts
        try:
            height, width = img_ori.shape[0:2]
        except:
            print(self.imgPathList[index])
            sys.exit('cannot get height,width')

        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('can not resize')

        # process img,set channel,and totensor
        # if 3 == self.channel:
        #     resized_im = resized_im.astype('float') * 0.0039216
        # if 1 == self.channel:
        #     resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
        #     resized_im = resized_im[:,:,np.newaxis]
        # print(resized_im[0][0])
        resized_im = resized_im * 0.0039216
        # print(resized_im[0][0])
        # time.sleep(1000)
        resized_im = resized_im[:,:,np.newaxis]

        # print(resized_im.shape)
        # print(resized_im[0][0])
        # time.sleep(1000)

        resized_im = resized_im.transpose(2,0,1)
        # test = torch.FloatTensor(resized_im)
        # print(resized_im.shape)
        # print(test.shape)
        # print(resized_im[0][0][0])
        # print(resized_im[0][0][1])
        # print(test[0][0][0])
        # print(test[0][0][1])

        eyePts = pts_ori[0:11,:].reshape((-1))
        mouthPts= pts_ori[11:,:].reshape((-1))
        # print(eyePts)
        # print(torch.FloatTensor([eyePts]).clamp_(min=0,max=1).shape)
        # time.sleep(1000)
        # print('a',type(resized_im))

        return resized_im, torch.FloatTensor([eyePts]), torch.FloatTensor([mouthPts])

class DatasetOnetOnLine(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 记录包含21点的图片名，和对应的人头序号。
        self.imgPathList21 = []
        self.labelPathList21 = []

        # 记录包含13点的图片名，和对应的人头序号。
        self.imgPathList13 = []
        self.labelPathList13 = []

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # # 检查是否包含错误样本
        # for i,imgPath in tqdm(enumerate(self.fileList)):

        #     filename = os.path.splitext(imgPath)[0]
        #     ptsPath = filename + '.pts'

        #     if os.path.exists(ptsPath):
        #         if isValidPts(ptsPath):
        #             continue
        #         else:
        #             self.fileList.pop(i)
        #             with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/errorSamplesAlign.txt','a') as f:
        #                 f.write(imgPath)
        #                 f.write('\n')
        #     else:
        #         self.fileList.pop(i)
        #         with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/errorSamplesAlign.txt','a') as f:
        #             f.write(imgPath)
        #             f.write('\n')

        print('After check have {} imgs'.format(len(self.fileList)))

        # split to 21 and 13 List, and check
        for imgPath in tqdm(self.fileList):
            filename = os.path.splitext(imgPath)[0]
            ptsPath = filename + '.pts'
            if os.path.exists(ptsPath):
                
                ptsList = getPts(ptsPath)
                npointsList = getnpoints(ptsPath)

                assert len(ptsList) == len(npointsList), 'something is wrong'

                for i,npoints in enumerate(npointsList):

                    if 21 == npoints:
                        self.imgPathList21.append(imgPath)
                        self.labelPathList21.append(i)
                    elif 13 == npoints:
                        self.imgPathList13.append(imgPath)
                        self.labelPathList13.append(i)
                    else:
                        sys.exit('{} have wrong num of points'.format(imgPath))
                # except:
                #     continue


        assert len(self.imgPathList21) == len(self.labelPathList21)
        assert len(self.imgPathList13) == len(self.labelPathList13)


        print('len(self.imgPathList21):',len(self.imgPathList21))
        print('len(self.imgPathList13):',len(self.imgPathList13))

        if 'eye' == self.target:
            self.imgPathList = self.imgPathList21 + self.imgPathList13
            self.labelPathList = self.labelPathList21 + self.labelPathList13
        if 'mouth' == self.target:
            self.imgPathList = self.imgPathList21
            self.labelPathList = self.labelPathList21

        print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[self.labelPathList[index]]


        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        pts = pts[0:11]
        # randomaug img and pts

        # try:
        img_ori, pts_ori = randomAug_Onet_pts(img,pts)
        # except:
        #     print(img.shape)
        #     print(pts)
        #     sys.exit('randomAug_pts:{} have wrong'.format(imgFile))

        # img_ori, pts_ori = randomAug_pts(img,pts)

        # process pts
        height, width = img_ori.shape[0:2]
        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('some wrong in dataset')

        # process img,set channel,and totensor
        # if 3 == self.channel:
        #     resized_im = resized_im.astype('float') * 0.0039216
        # if 1 == self.channel:
        #     resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
        #     resized_im = resized_im[:,:,np.newaxis]
        # print(resized_im[0][0])
        resized_im = resized_im * 0.0039216
        # print(resized_im[0][0])
        # time.sleep(1000)
        resized_im = resized_im[:,:,np.newaxis]

        # print(resized_im.shape)
        # print(resized_im[0][0])
        # time.sleep(1000)

        resized_im = resized_im.transpose(2,0,1)
        # test = torch.FloatTensor(resized_im)
        # print(resized_im.shape)
        # print(test.shape)
        # print(resized_im[0][0][0])
        # print(resized_im[0][0][1])
        # print(test[0][0][0])
        # print(test[0][0][1])

        eyePts = pts_ori[0:11,:].reshape((-1))
        # mouthPts= pts_ori[11:,:].reshape((-1))
        # print(eyePts)
        # print(torch.FloatTensor([eyePts]).clamp_(min=0,max=1).shape)
        # time.sleep(1000)
        # print('a',type(resized_im))

        return resized_im, torch.FloatTensor([eyePts])

class DatasetOnetOffLine(torchDataset):

    def __init__(self, txtList=None, size=64, channel=1, isTrain='train'):

        self.size = size
        self.channel = channel
        self.isTrain = isTrain

        # 获取全部图片
        self.fileList = []
        self.labeList = []

        for i in range(len(txtList)):
            with open(txtList[i],'r') as f:
                lines = f.readlines()
                for i in range(len(lines)-1):
                    line = lines[i]
                    annotations = line.strip().split(' ')


                    pts = np.array(annotations[1:]).reshape((-1,2)).astype(np.float16).tolist()
                    self.fileList.append(annotations[0])
                    self.labeList.append(pts)

        # print('before')
        # print(self.labeList[40490])
        # print(self.fileList[40490])

        # 打乱已获取到的图片顺序
        random.seed(666)
        random.shuffle(self.fileList)
        random.seed(666)
        random.shuffle(self.labeList)

        self.labeList = np.array(self.labeList)

        # print('after')
        # print(self.labeList.dtype)
        # print(self.labeList[99262])
        # print(self.fileList[99262])

        # time.sleep(1000)


        N = len(self.fileList)

        if 'train' == self.isTrain:
            self.fileList = self.fileList[0:int(0.8*N)]
            self.labeList = self.labeList[0:int(0.8*N),]
        else:
            self.fileList = self.fileList[int(0.8*N):]
            self.labeList = self.labeList[int(0.8*N):,]

        assert len(self.fileList) == len(self.labeList), "wrong number between files and labels"

        print('{} have {} imgs'.format(self.isTrain,len(self.fileList)))
        print('self.labeList have shape {}'.format(self.labeList.shape))

    def __len__(self):

        return len(self.fileList)

    def __getitem__(self, index):

        # 获取对应文件名
        imgFile = self.fileList[index]
        img_ori = cv2.imread(imgFile,0)

        pts = self.labeList[index]

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('some wrong in dataset')

        # process img,set channel,and totensor
        # if 3 == self.channel:
        #     resized_im = resized_im.astype('float') * 0.0039216
        # if 1 == self.channel:
        #     resized_im = cv2.cvtColor(resized_im,cv2.COLOR_BGR2GRAY).astype('float') * 0.0039216
        #     resized_im = resized_im[:,:,np.newaxis]
        # print(resized_im[0][0])
        resized_im = resized_im * 0.0039216
        # print(resized_im[0][0])
        # time.sleep(1000)
        resized_im = resized_im[:,:,np.newaxis]

        resized_im = resized_im.transpose(2,0,1)

        return resized_im, torch.FloatTensor([pts])

class DatasetOnetClassOnLine(torchDataset):

    def __init__(self, imgDir=None, size=64, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 记录包含21点的图片名，和对应的人头序号。
        self.imgPathList21 = []
        self.labelPathList21 = []

        # 记录包含13点的图片名，和对应的人头序号。
        self.imgPathList13 = []
        self.labelPathList13 = []

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)


        self.fileList = filter(isValidImg,self.fileList)
        self.fileList = list(self.fileList)
        print('After check have {} imgs'.format(len(self.fileList)))

        # split to 21 and 13 List, and check
        for imgPath in self.fileList:
            filename = os.path.splitext(imgPath)[0]
            ptsPath = filename + '.pts'
            if os.path.exists(ptsPath):
                
                ptsList = getPts(ptsPath)
                npointsList = getnpoints(ptsPath)

                assert len(ptsList) == len(npointsList), 'something is wrong'

                for i,npoints in enumerate(npointsList):

                    if 21 == npoints:
                        self.imgPathList21.append(imgPath)
                        self.labelPathList21.append(i)
                    elif 13 == npoints:
                        self.imgPathList13.append(imgPath)
                        self.labelPathList13.append(i)
                    else:
                        sys.exit('{} have wrong num of points'.format(imgPath))
                # except:
                #     continue


        assert len(self.imgPathList21) == len(self.labelPathList21)
        assert len(self.imgPathList13) == len(self.labelPathList13)


        print('len(self.imgPathList21):',len(self.imgPathList21))
        print('len(self.imgPathList13):',len(self.imgPathList13))

        if 'eye' == self.target:
            self.imgPathList = self.imgPathList21 + self.imgPathList13
            self.labelPathList = self.labelPathList21 + self.labelPathList13
        if 'mouth' == self.target:
            self.imgPathList = self.imgPathList21
            self.labelPathList = self.labelPathList21

        print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[self.labelPathList[index]]

        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        pts_11 = pts[0:11,]

        minx,miny = pts_11.min(axis=0)
        maxx,maxy = pts_11.max(axis=0)

        box = [[minx,miny,maxx,maxy]]
        box_ori = copy.deepcopy(box)
        # get pos samples
        if(random.random() < 0.3):

            img_new, box_new = randomAugOnet_box(img,box)
            height, width = img_new.shape[0:2]

            cropped_im = img_new
            resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

            label = 0

        # get neg samples,use iou
        else:
            minx,miny = pts.min(axis=0)
            maxx,maxy = pts.max(axis=0)
            box = [[minx,miny,maxx,maxy]]
            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 1

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
            assert width > 100, print(self.imgPathList[index])
            assert height > 100, print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
            resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetOnetGazeOnLine(torchDataset):

    def __init__(self, imgDir=None, size=64, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # 获取想要的方向
        self.fileList = filter(fillterWrongDirect,self.fileList)
        self.fileList = list(self.fileList)
        print('After filterd have {} imgs'.format(len(self.fileList)))
        self.angleList = [0] * 3
        for singlefile in tqdm(self.fileList):
            if('Left' in singlefile or 'left' in singlefile):
                self.angleList[0] += 1
            elif('Right' in singlefile or 'right' in singlefile):
                self.angleList[2] += 1
            else:
                self.angleList[1] += 1
        print('self.angleList:',self.angleList)
        # 检修问题样本
        self.fileList = filter(easyCheckPts,self.fileList)
        self.fileList = list(self.fileList)
        print('After checked have {} imgs'.format(len(self.fileList)))


        N = len(self.fileList)
        if 'train' == isTrain:
            self.imgPathList = self.fileList[:int(0.8*N)]
        else:
            self.imgPathList = self.fileList[int(0.8*N):]

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        ptsFile = os.path.splitext(imgFile)[0] + '.pts'

        # 获取图片和标签
        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        if('Left' in imgFile or 'Left' in imgFile):
            label = 0
        elif('Middle' in imgFile or 'Centre' in imgFile or 'Center' in imgFile):
            label = 1
        elif('Right' in imgFile or 'right' in imgFile):
            label = 2
        else:
            print(imgFile)
            sys.exit('label is wrong')

        pts = easyGetPts(ptsFile)

        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
        pts = revisePts(pts,width,height)

        # pts_11 = pts[0:11,]

        # minx,miny = pts_11.min(axis=0)
        # maxx,maxy = pts_11.max(axis=0)

        pts_4 = pts[1:5,]

        minx,miny = pts_4.min(axis=0)
        maxx,maxy = pts_4.max(axis=0)

        box = [[minx,miny,maxx,maxy]]
        box_ori = copy.deepcopy(box)

        img_new, box_new = randomAugOnet_box(img,box)

        # # 随机水平对称
        # if random.random() < 0.3:
        #     img_new = cv2.flip(img_new,1)
        #     label = 2 - label

        height, width = img_new.shape[0:2]

        cropped_im = img_new
        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetOnetGazeOnLineMultiLabel(torchDataset):

    def __init__(self, imgDir=None, size=64, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # 获取想要的方向
        self.fileList = filter(fillterWrongDirect,self.fileList)
        self.fileList = list(self.fileList)
        print('After filterd have {} imgs'.format(len(self.fileList)))
        self.angleList = [0] * 3
        for singlefile in tqdm(self.fileList):
            if('Left' in singlefile or 'left' in singlefile):
                self.angleList[0] += 1
            elif('Right' in singlefile or 'right' in singlefile):
                self.angleList[2] += 1
            else:
                self.angleList[1] += 1
        print('self.angleList:',self.angleList)
        # 检修问题样本
        self.fileList = filter(easyCheckPts,self.fileList)
        self.fileList = list(self.fileList)
        print('After checked have {} imgs'.format(len(self.fileList)))


        N = len(self.fileList)
        if 'train' == isTrain:
            self.imgPathList = self.fileList[:int(0.8*N)]
        else:
            self.imgPathList = self.fileList[int(0.8*N):]

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        ptsFile = os.path.splitext(imgFile)[0] + '.pts'

        # 获取图片和标签
        img = cv2.imread(imgFile,0)
        try:
            height,width = img.shape[0:2]
        except:
            print('{} have wrong'.format(imgFile))
            sys.exit()
        label = [0] * 5

        if('Left' in imgFile or 'left' in imgFile):
            label[0] = 1
        if('Middle' in imgFile or 'Centre' in imgFile or 'Center' in imgFile):
            label[1] = 1
        if('Right' in imgFile or 'right' in imgFile):
            label[2] = 1
        if('Up' in imgFile or 'up' in imgFile):
            label[3] = 1
        if('down' in imgFile or 'Down' in imgFile):
            label[4] = 1

        if(label[0] == 0 and label[1] == 0 and label[2] == 0):
            label[1] = 0

        pts = easyGetPts(ptsFile)

        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
        pts = revisePts(pts,width,height)

        # pts_11 = pts[0:11,]

        # minx,miny = pts_11.min(axis=0)
        # maxx,maxy = pts_11.max(axis=0)

        pts_4 = pts[1:5,]

        minx,miny = pts_4.min(axis=0)
        maxx,maxy = pts_4.max(axis=0)

        box = [[minx,miny,maxx,maxy]]
        box_ori = copy.deepcopy(box)

        img_new, box_new = randomAugOnet_box(img,box)

        # # 随机水平对称
        # if random.random() < 0.3:
        #     img_new = cv2.flip(img_new,1)
        #     label = 2 - label

        height, width = img_new.shape[0:2]

        cropped_im = img_new
        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor(label).type(torch.FloatTensor)

class DatasetOnetClassOffLine(torchDataset):

    def __init__(self,txtDir,size=64,channle=1,isTrain='train'):

        self.txtList = getlist(txtDir,'.txt')
        self.imgList = []
        self.size = size

        for txtFile in self.txtList:
            with open(txtFile,'r') as f:
                lines = f.readlines()
            print("{} have {} imgs".format(txtFile,len(lines)))
            for line in lines:
                self.imgList.append(line.strip()[:-2])

        random.seed(0)
        random.shuffle(self.imgList)

        N = len(self.imgList)

        if('train' == isTrain):
            self.imgList = self.imgList[0:int(N*0.8)]

        if('val' == isTrain):
            self.imgList = self.imgList[int(N*0.8):]

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgList[index],0)
        imgReshape = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        imgReshape = imgReshape[np.newaxis,]

        if('positive' in self.imgList[index]):
            label = 0
        elif('negtive' in self.imgList[index]):
            label = 1
        else:
            print(self.imgList[index])
            sys.exit('wrong label')

        return imgReshape,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetEmotion(torchDataset):

    def __init__(self, imgDir='/mnt/workspace/jiangtao/dataset/train/emotion/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml').replace('/img_','/xml_')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.95)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.7):

            img_new, box_new = randomAug_box(img,box)

            height, width = img_new.shape[0:2]

            x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
            w = x2 - x1
            h = y2 - y1

            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

            if('img_0' in self.imgPathList[index]):
                label = 0
            if('img_1' in self.imgPathList[index]):
                label = 1

        #get neg samples
        # else:
        #     height, width = img.shape[0:2]
        #     x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
        #     label = 2

        #     if (x1 > 128) and (y1 > 128):
        #         x1 = random.randint(int(x1/2),int(x1))
        #         y1 = random.randint(int(y1/2),int(y1))
        #         cropped_im = img[0:int(y1), 0:int(y2)]
        #         resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        #         resized_im = resized_im[np.newaxis,]
        #     else:
        #         resized_im = np.random.rand(1,128,128)

        # get neg samples,use iou
        else:
            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 2

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
            # assert width > 100,print(self.imgPathList[index])
            # assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width/2))
            y1 = random.randint(0,int(height/2))
            x2 = random.randint(x1+1,width)
            y2 = random.randint(y1+1,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
            resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label])

class DatasetCover(torchDataset):

    def __init__(self, imgDir=[], size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
        random.shuffle(self.fileList)
        print(len(self.fileList))
        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            
            # img_3 dont need xml,so add imgPath to labelPathList
            if('img_3' in imgPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(imgPath)
                continue

            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        if('img_3' in self.imgPathList[index]):
            img = cv2.imread(self.imgPathList[index],0)
            height, width = img.shape[0:2]
            assert width > 100,print(self.imgPathList[index])
            assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)
            box = [[x1,y1,x2,y2]]

        else:
            img = cv2.imread(self.imgPathList[index],0)
            box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.65):
            try:
                img_new, box_new = randomAug_box(img,box)
            except:
                print(self.imgPathList[index])
                print(box)
                sys.exit('img or label is wrong')

            height, width = img_new.shape[0:2]

            x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
            w = x2 - x1
            h = y2 - y1

            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

            if('img_0' in self.imgPathList[index]):
                label = 0
            if('img_1' in self.imgPathList[index]):
                label = 1
            if('img_2' in self.imgPathList[index]):
                label = 2
            if('img_3' in self.imgPathList[index]):
                label = 3

        # get neg samples,use left up
        # else:
        #     height, width = img.shape[0:2]
        #     x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
        #     label = 3

        #     if (x1 > 128) and (y1 > 128):
        #         x1 = random.randint(int(x1/2),int(x1))
        #         y1 = random.randint(int(y1/2),int(y1))
        #         cropped_im = img[0:int(y1), 0:int(y2)]
        #         resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        #         resized_im = resized_im[np.newaxis,]
        #     else:
        #         resized_im = np.random.rand(1,128,128)

        # get neg samples,use iou
        else:
            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 3

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
            assert width > 100,print(self.imgPathList[index])
            assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
            resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetAngle(torchDataset):

    def __init__(self, imgDir='/home/jiangtao/workspace/dataset/train/angle/DSM/unmask/', size=128, imgChannel=1, isTrain='train'):

        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        self.fileList.sort()
        random.seed(612)
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        self.classes_for_all_imgs = []
        self.numbers_for_all_classes = [0,0,0,0,0]

        for imgPath in self.imgPathList:

            if ('/largeleft/') in imgPath:
                class_id = 0
            elif ('/left/') in imgPath:
                class_id = 1
            elif ('/middle/') in imgPath:
                class_id = 2
            elif ('/right/') in imgPath:
                class_id = 3
            elif ('/largeright/') in imgPath:
                class_id = 4
            else:
                print(imgPath)
                # time.sleep(1000)

            self.numbers_for_all_classes[int(class_id)] += 1
            self.classes_for_all_imgs.append(class_id)
            
        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.75):

            img_new, box_new = randomAug_box(img,box)

            height, width = img_new.shape[0:2]

            x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
            w = x2 - x1
            h = y2 - y1

            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

            # print(self.imgPathList[index])
            # time.sleep(1000)

            if('/largeleft/' in self.imgPathList[index]):
                label = 0     

            elif('/left/' in self.imgPathList[index]):
                label = 1

            elif('/middle/' in self.imgPathList[index]):
                label = 2

            elif('/right/' in self.imgPathList[index]):
                label = 3

            elif('/largeright/' in self.imgPathList[index]):
                label = 4

            else:
                print(self.imgPathList[index])
                # time.sleep(1000)

                
        # get neg samples,use iou
        else:
            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 5

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
            assert width > 100,print(self.imgPathList[index])
            assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+1,width)
            y2 = random.randint(y1+1,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
            resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label])

    @property
    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

    @property
    def get_numbers_for_all_classes(self):
        return self.numbers_for_all_classes

class DatasetFaceArea(torchDataset):

    def __init__(self, imgDir='/home/jiangtao/workspace/dataset/train/detect/img_DSM/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.7):
            
            # face is complete,so label is 1
            label = 1
            
            img_new, box_new = randomAug_box(img,box)
            height, width = img_new.shape[0:2]
            x1 = int(box_new[0][0])
            y1 = int(box_new[0][1])
            x2 = int(box_new[0][2])
            y2 = int(box_new[0][3])
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            w = x2 - x1
            h = y2 - y1

            if(random.random() > 0.65):
                label -= 0.25
                img_new = mastOnImage(img_new,x1,y1,x_center,y_center)

            if(random.random() > 0.65):
                label -= 0.25
                img_new = mastOnImage(img_new,x_center,y1,x2,y_center)

            if(random.random() > 0.65):
                label -= 0.25
                img_new = mastOnImage(img_new,x1,y_center,x_center,y2)

            if(random.random() > 0.65):
                label -= 0.25
                img_new = mastOnImage(img_new,x_center,y_center,x2,y2)

            # random crop
            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

        #get neg samples
        else:
            height, width = img.shape[0:2]
            x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 0

            if (x1 > 128) and (y1 > 128):
                x1 = random.randint(int(x1/2),int(x1))
                y1 = random.randint(int(y1/2),int(y1))
                cropped_im = img[0:int(y1), 0:int(y2)]
                resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
                resized_im = resized_im[np.newaxis,]
            else:
                resized_im = np.random.randint(255,size=(1,128,128)).astype('float')*0.0039216

        return resized_im,torch.FloatTensor([label])

class DatasetFaceArea_V2(torchDataset):

    def __init__(self, imgDir='/home/jiangtao/workspace/dataset/train/detect/img_DSM/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.7):
            
            # face is complete,so label is 1
            label = 1
            
            img_new, box_new = randomAug_box(img,box)
            height, width = img_new.shape[0:2]
            x1 = int(box_new[0][0])
            y1 = int(box_new[0][1])
            x2 = int(box_new[0][2])
            y2 = int(box_new[0][3])
            w = x2 - x1
            h = y2 - y1

            dice = random.random()
            
            if(0<=dice<0.2):
                scale = random.uniform(0,0.5)
                label -= scale
                cut = int(scale * h + y1)
                img_new = img_new[cut:,:]
                height, width = img_new.shape[0:2]
                y1 = 0
                y2 -= cut

            elif(0.2<=dice<0.4):
                scale = random.uniform(0,0.5)
                label -= scale
                cut = int(y2 - scale * h)
                img_new = img_new[:cut,:]
                height, width = img_new.shape[0:2]
                y2 = height

            elif(0.4<=dice<0.6):
                scale = random.uniform(0,0.5)
                label -= scale
                cut = int(x1 + scale * w)
                img_new = img_new[:,cut:]
                height, width = img_new.shape[0:2]
                x1 = 0 
                x2 -= cut

            elif(0.6<=dice<0.8):
                scale = random.uniform(0,0.5)
                label -= scale
                cut = int(x2 - scale * w)
                img_new = img_new[:,:cut]
                height, width = img_new.shape[0:2]
                x2 = width
            
            else:
                scale = 0

            # random crop
            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

        #get neg samples
        else:
            height, width = img.shape[0:2]
            x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 0

            if (x1 > 128) and (y1 > 128):
                x1 = random.randint(int(x1/2),int(x1))
                y1 = random.randint(int(y1/2),int(y1))
                cropped_im = img[0:int(y1), 0:int(y2)]
                resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
                resized_im = resized_im[np.newaxis,]
            else:
                resized_im = np.random.randint(255,size=(1,128,128)).astype('float')*0.0039216

        return resized_im,torch.FloatTensor([label])

class DatasetDetect_v2(torchDataset):

    def __init__(self, imgDir='/home/jiangtao/workspace/dataset/train/detect/img_DSM/0/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)


        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        img_new, box_new = randomAug_box(img,box)

        height, width = img_new.shape[0:2]

        x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
        w = x2 - x1
        h = y2 - y1

        if random.random() < 0.5:
            delta_x1 = np.random.randint(0,int(w * 0.5))
            delta_y1 = np.random.randint(0,int(h * 0.5))
            delta_x2 = np.random.randint(0,int(w * 0.5)) 
            delta_y2 = np.random.randint(0,int(h * 0.5)) 

        else:
            delta_x1 = np.random.randint(int(w * 0.5), int(w * 10.0))
            delta_y1 = np.random.randint(int(h * 0.5), int(h * 10.0))
            delta_x2 = np.random.randint(int(w * 0.5), int(w * 10.0)) 
            delta_y2 = np.random.randint(int(h * 0.5), int(h * 10.0)) 

        nx1 = max(x1 - delta_x1,0)
        ny1 = max(y1 - delta_y1,0)
        nx2 = min(x2 + delta_x2,width)
        ny2 = min(y2 + delta_y2,height)

        # 将点归一化到裁剪区域中
        x1 = (x1 - nx1) * 128 / (nx2 - nx1)
        y1 = (y1 - ny1) * 128 / (ny2 - ny1)
    
        x1 = x1 / 128.0000000000
        y1 = y1 / 128.0000000000

        x2 = (x2 - nx1) * 128 / (nx2 - nx1)
        y2 = (y2 - ny1) * 128 / (ny2 - ny1)
    
        x2 = x2 / 128.0000000000
        y2 = y2 / 128.0000000000

        cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
        resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        labels_out = torch.zeros((1, 6))
        labels_out[0][1] = 0
        labels_out[:, 2:] = torch.FloatTensor([x1, y1, x2, y2])
        labels_out[:, 2:] = xyxy2xywh(labels_out[:, 2:])
        
        return torch.from_numpy(resized_im),labels_out

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)

class DatasetDetect_v3(torchDataset):

    def __init__(self, imgDir='/home/jiangtao/workspace/dataset/train/detect/img_DSM/0/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)


        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]


        self.height = 32
        self.width = 32
        self.downscale = 4
        self.sigma = 2.65
        
        self.grid_x = np.tile(np.arange(self.width), reps=(self.height, 1))
        self.grid_y = np.tile(np.arange(self.height), reps=(self.width, 1)).transpose()

        print(len(self.imgPathList))
        print(len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        img_new, box_new = randomAug_box(img,box)

        height, width = img_new.shape[0:2]

        x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
        w = x2 - x1
        h = y2 - y1

        if random.random() < 0.5:
            delta_x1 = np.random.randint(0,int(w * 0.5))
            delta_y1 = np.random.randint(0,int(h * 0.5))
            delta_x2 = np.random.randint(0,int(w * 0.5)) 
            delta_y2 = np.random.randint(0,int(h * 0.5)) 

        else:
            delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.5))
            delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.5))
            delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.5)) 
            delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.5)) 

        nx1 = max(x1 - delta_x1,0)
        ny1 = max(y1 - delta_y1,0)
        nx2 = min(x2 + delta_x2,width)
        ny2 = min(y2 + delta_y2,height)

        x1 = (x1 - nx1) * 128.0 / (nx2 - nx1)
        y1 = (y1 - ny1) * 128.0 / (ny2 - ny1)
    
        x2 = (x2 - nx1) * 128.0 / (nx2 - nx1)
        y2 = (y2 - ny1) * 128.0 / (ny2 - ny1)


        cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
        resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        resized_im = resized_im[np.newaxis,]


        res = np.zeros([5, self.height, self.width], dtype=np.float32)
        
        left,top,right,bottom = int(x1/self.downscale),int(y1/self.downscale),int(x2/self.downscale),int(y2/self.downscale)
        x = (left + right) // 2
        y = (top + bottom) // 2
        grid_dist = (self.grid_x - x) ** 2 + (self.grid_y - y) ** 2
        heatmap = np.exp(-0.5 * grid_dist / self.sigma ** 2)
        res[0] = np.maximum(heatmap, res[0])

        original_x = (x1 + x2) / 2
        original_y = (y1 + y2) / 2
        res[1][y, x] = original_x / self.downscale - x
        res[2][y, x] = original_y / self.downscale - y

        width = x2 - x1
        height = y2 - y1
        res[3][y, x] = np.log(width + 1e-4)
        res[4][y, x] = np.log(height + 1e-4)

        return resized_im,res

class DatasetRecog(torchDataset):

    def __init__(self, imgDir='/jiangtao2/dataset/train/recognition/', size=128, imgChannel=1, isTrain='train'):
        
        self.size = size

        # get imgpath
        self.fileList = getlist(imgDir,'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []
        self.idList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)
                self.idList.append(osp.basename(osp.dirname(imgPath)))

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]
            self.idList = self.idList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]
            self.idList = self.idList[nTrain:]

        # print(len(self.imgPathList))
        # print(len(self.labelPathList))
        # print(len(self.idList))

        # print(self.imgPathList[0])
        # print(self.labelPathList[0])
        # print(self.idList[0])

        self.meteData = []
        for i in range(len(self.imgPathList)):
            self.meteData.append([self.imgPathList[i], self.labelPathList[i], self.idList[i]])

        # print(self.meteData[5][0])
        # print(self.meteData[5][1])
        # print(self.meteData[5][2])

        # relabel
        pid_set = set([i[2] for i in self.meteData])
        self.pids = sorted(list(pid_set))
        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        # print(self.pid_dict)
        # print(type(self.pid_dict))
        # print(len(pid_set))
        for i in range(len(self.imgPathList)):
            self.meteData[i][2] = self.pid_dict[self.meteData[i][2]]

        # print(self.meteData[5][0])
        # print(self.meteData[5][1])
        # print(self.meteData[5][2])

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.meteData[index][0],0)
        box = getbox(self.meteData[index][1])
        pid = self.meteData[index][2]

        img_new, box_new = randomAug_box(img,box)

        height, width = img_new.shape[0:2]

        x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
        w = x2 - x1
        h = y2 - y1
        
        dice = random.random()
        
        # reid扩充，最多包含半张人脸的背景
        delta_x1 = np.random.randint(int(w * 0.), int(w * 0.25))
        delta_y1 = np.random.randint(int(h * 0.), int(h * 0.25))
        delta_x2 = np.random.randint(int(w * 0.), int(w * 0.25)) 
        delta_y2 = np.random.randint(int(h * 0.), int(h * 0.25)) 

        nx1 = max(x1 - delta_x1,0)
        ny1 = max(y1 - delta_y1,0)
        nx2 = min(x2 + delta_x2,width)
        ny2 = min(y2 + delta_y2,height)

        # 将点归一化到裁剪区域中
        x1 = (x1 - nx1) * 128 / (nx2 - nx1)
        y1 = (y1 - ny1) * 128 / (ny2 - ny1)
    
        x1 = x1 / 128.0000000000
        y1 = y1 / 128.0000000000

        x2 = (x2 - nx1) * 128 / (nx2 - nx1)
        y2 = (y2 - ny1) * 128 / (ny2 - ny1)
    
        x2 = x2 / 128.0000000000
        y2 = y2 / 128.0000000000

        cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
        resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return torch.FloatTensor(resized_im).type(torch.FloatTensor),torch.FloatTensor([x1, y1, x2, y2]).clamp_(min=0,max=1),pid

class DatasetDMSGender(torchDataset):

    def __init__(self, imgDir, imgTxt, size=128, imgChannel=1, isTrain='train'):

        self.size = size

        # get imgpath
        self.fileList = []
        self.imgTxt = os.path.join(imgDir,imgTxt)
        with open(self.imgTxt,'r') as f:
            lines = f.readlines()
        for line in lines:
            self.fileList.append(imgDir + line.strip())
        
        self.imgPathList = []
        self.labelPathList = []
        # get labelpath
        for imgPath in tqdm(self.fileList):

            xmlPath = imgPath.replace('.jpg','.xml')
            '''去除三岁以下儿童，因为性别特征不明显，可能会对网络造成负面影响'''
            age = getage(xmlPath)
            if(age <= 3):
                continue
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self,index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])
        gender = getgender(self.labelPathList[index])

        # get new img and new box
        img_new, box_new = randomAug_box(img,box)
        # # get pos samples
        # try:
        #     img_new, box_new = randomAug_box(img,box)
        # except:
        #     print(self.imgPathList[index])
        #     print(box)
        #     sys.exit('img or label is wrong')

        ret = randomAug_boxV2(img_new,box_new,0.15)

        if(ret[0] == False):
            sys.exit('{} have problem:{}'.format(self.imgPathList[index],ret[1]))
        else:
            cropped_im = ret[1]

        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return resized_im, torch.FloatTensor([gender]).type(torch.FloatTensor)

class DatasetDMSAge(torchDataset):

    def __init__(self, imgDir=[], size=128, imgChannel=1, isTrain='train'):

        self.size = size

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:

            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self,index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])
        age = getage(self.labelPathList[index])

        bins = np.array(range(0, 78, 3))
        binned_age = np.digitize(age, bins) - 1

        # get new img and new box
        img_new, box_new = randomAug_box(img,box)

        ret = randomAug_boxV2(img_new,box_new,0.15)

        if(ret[0] == False):
            sys.exit('{} have problem:{}'.format(self.imgPathList[index],ret[1]))
        else:
            cropped_im = ret[1]

        resized_im = cv2.resize(cropped_im, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        resized_im = resized_im[np.newaxis,]

        return resized_im, torch.FloatTensor([binned_age]).type(torch.FloatTensor), torch.FloatTensor([age]).type(torch.FloatTensor)

class DatasetBinaryFace(torchDataset):

    def __init__(self, imgDir=[], size=128, imgChannel=1, isTrain='train'):

        self.size = size

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
        random.shuffle(self.fileList)
        print(len(self.fileList))
        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            
            # img_3 dont need xml,so add imgPath to labelPathList
            if('img_3' in imgPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(imgPath)
                continue

            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        if('img_3' in self.imgPathList[index]):
            img = cv2.imread(self.imgPathList[index],0)
            height, width = img.shape[0:2]
            assert width > 100,print(self.imgPathList[index])
            assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)
            box = [[x1,y1,x2,y2]]

        else:
            img = cv2.imread(self.imgPathList[index],0)
            box = getbox(self.labelPathList[index])

        # get pos samples
        if(random.random() < 0.40):
            try:
                img_new, box_new = randomAug_box(img,box)
            except:
                print(self.imgPathList[index])
                print(box)
                sys.exit('img or label is wrong')

            height, width = img_new.shape[0:2]

            x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
            w = x2 - x1
            h = y2 - y1

            assert w >= 50,' {} have wrong'.format(self.imgPathList[index])
            assert h >= 50,' {} have wrong'.format(self.imgPathList[index])

            if random.random() < 0.5:
                delta_x1 = np.random.randint(0,int(w * 0.5))
                delta_y1 = np.random.randint(0,int(h * 0.5))
                delta_x2 = np.random.randint(0,int(w * 0.5)) 
                delta_y2 = np.random.randint(0,int(h * 0.5)) 

            else:
                delta_x1 = np.random.randint(int(w * 0.5), int(w * 1.0))
                delta_y1 = np.random.randint(int(h * 0.5), int(h * 1.0))
                delta_x2 = np.random.randint(int(w * 0.5), int(w * 1.0)) 
                delta_y2 = np.random.randint(int(h * 0.5), int(h * 1.0)) 

            nx1 = max(x1 - delta_x1,0)
            ny1 = max(y1 - delta_y1,0)
            nx2 = min(x2 + delta_x2,width)
            ny2 = min(y2 + delta_y2,height)

            assert ny2 > ny1 + 50,' {} have wrong y1 {} y2 {} ny1 {} ny2 {} delta_y1 {} delta_y2 {}'.format(self.imgPathList[index],y1,y2,ny1,ny2,delta_y1,delta_y2)
            assert nx2 > nx1 + 50,' {} have wrong x1 {} x2 {} nx1 {} nx2 {} delta_x1 {} delta_x2 {}'.format(self.imgPathList[index],x1,x2,nx1,nx2,delta_x1,delta_x2)

            # 将点归一化到裁剪区域中
            x1 = (x1 - nx1) * 128 / (nx2 - nx1)
            y1 = (y1 - ny1) * 128 / (ny2 - ny1)
        
            x1 = x1 / 128.0000000000
            y1 = y1 / 128.0000000000

            x2 = (x2 - nx1) * 128 / (nx2 - nx1)
            y2 = (y2 - ny1) * 128 / (ny2 - ny1)
        
            x2 = x2 / 128.0000000000
            y2 = y2 / 128.0000000000

            cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

            resized_im = resized_im[np.newaxis,]

            if('img_0' in self.imgPathList[index]):
                label = 0
            if('img_1' in self.imgPathList[index]):
                label = 0
            if('img_2' in self.imgPathList[index]):
                label = 0

            if('img_3' in self.imgPathList[index]):
                label = 1

        # get neg samples,use left up
        # else:
        #     height, width = img.shape[0:2]
        #     x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
        #     label = 3

        #     if (x1 > 128) and (y1 > 128):
        #         x1 = random.randint(int(x1/2),int(x1))
        #         y1 = random.randint(int(y1/2),int(y1))
        #         cropped_im = img[0:int(y1), 0:int(y2)]
        #         resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
        #         resized_im = resized_im[np.newaxis,]
        #     else:
        #         resized_im = np.random.rand(1,128,128)

        # get neg samples,use iou
        else:
            height, width = img.shape[0:2]
            minX,minY,maxX,maxY = box[0][0],box[0][1],box[0][2],box[0][3]
            label = 1

            img[int(minY):int(maxY),int(minX):int(maxX)] = img.mean()
            assert width > 100,print(self.imgPathList[index])
            assert height > 100,print(self.imgPathList[index])
            x1 = random.randint(0,int(width-100))
            y1 = random.randint(0,int(height-100))
            x2 = random.randint(x1+100,width)
            y2 = random.randint(y1+100,height)

            cropped_im = img[int(y1):int(y2), int(x1):int(x2)]
            resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216
            resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetGaze(torchDataset):

    def __init__(self, imgDir, size=128, imgChannel=1, isTrain='train'):

        self.size = size
        self.imgDir = imgDir

        # get imgpath
        dirs = os.listdir(self.imgDir)
        dirs = [osp.join(self.imgDir,dir) for dir in dirs]

        self.fileList = []
        for dir in dirs:
            fileList = getlist(dir,'.jpg')
            self.fileList.extend(fileList)
            print('{} have {} images'.format(dir,len(fileList)))
        print("{} have {} images".format(self.imgDir,len(self.fileList)))

        random.seed(666)
        random.shuffle(self.fileList)
        # print(len(self.fileList))

        self.imgPathList = []
        self.labelPathList = []

        # get labelpath
        for imgPath in self.fileList:
            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        N = len(self.imgPathList)
        if 'train' == isTrain:
            self.imgPathList = self.imgPathList[0:int(0.8*N)]
            self.labelPathList = self.labelPathList[0:int(0.8*N)]
        else:
            self.imgPathList = self.imgPathList[int(0.8*N):]
            self.labelPathList = self.labelPathList[int(0.8*N):]

        assert len(self.imgPathList) == len(self.labelPathList)

        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getbox(self.labelPathList[index])

        try:
            img_new, box_new = randomAug_Gaze(img,box)
        except:
            print(self.imgPathList[index])
            print(box)
            sys.exit('img or label is wrong')

        if('/eyeLeft' in self.imgPathList[index]):
            label = 0
        elif('/eyeRight' in self.imgPathList[index]):
            label = 2
        else:
            label = 1

        if random.random() > 0.5:
            img_new = cv2.flip(img_new,1)
            label = 2 - label

        resized_im = cv2.resize(img_new, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)

class DatasetAngleReg(torchDataset):

    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train', target='eye'):

        self.size = size
        self.channel = channel
        self.target = target

        # 统计到的符合条件的图片
        self.imgPathList = []
        self.labelPathList = []

        # 获取全部图片
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
            print('{} have {} imgs'.format(imgDir[i],len(getlist(imgDir[i],'.jpg'))))

        print('Total have {} imgs'.format(len(self.fileList)))

        # 打乱已获取到的图片顺序
        self.fileList.sort()
        random.seed(0)
        random.shuffle(self.fileList)

        # 检查是否包含错误样本
        # for i,imgPath in tqdm(enumerate(self.fileList)):

        #     filename = os.path.splitext(imgPath)[0]
        #     ptsPath = filename + '.pts'

        #     if os.path.exists(ptsPath):
        #         if isValidPts(ptsPath):
        #             continue
        #         else:
        #             self.fileList.pop(i)
        #             with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/errorSamplesAlign.txt','a') as f:
        #                 f.write(imgPath)
        #                 f.write('\n')
        #     else:
        #         self.fileList.pop(i)
        #         with open('/jiangtao2/code_with_git/MultiTaskOnFaceRebuild/errorSamplesAlign.txt','a') as f:
        #             f.write(imgPath)
        #             f.write('\n')

        # print('After check have {} imgs'.format(len(self.fileList)))






        nTrain = int(len(self.fileList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.fileList[0:nTrain]
            self.labelPathList = self.fileList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.fileList[nTrain:]
            self.labelPathList = self.fileList[nTrain:]

        print('isTrain:',isTrain)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

        # print('len(self.imgPathList):',len(self.imgPathList))

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

        # read img and pts, and set channel to 3
        # if have box,then read box
        # img = cv2.imread(self.imgPathList[index],1)

        # 获取对应文件名
        imgFile = self.imgPathList[index]
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        img = cv2.imread(imgFile,0)
        height,width = img.shape[0:2]

        ptsList = getPts(ptsFile)
        pts = ptsList[0]


        # box = np.zeros((1,4))
        if(21 == pts.shape[0]):
            pts = pts21to19(pts)
            pts = revisePts(pts,width,height)

        # randomaug img and pts
        try:
            img_ori, pts_ori = randomAug_pts(img,pts)
        except:
            print(img_ori.shape)
            print(pts_ori)
            sys.exit('randomAug_pts:{} have wrong'.format(imgFile))

        # get label
        label = imgFile.split('/')[-2]
        label = float(label) / 9
        if 'left' in imgFile:
            label = -label

        # 随机对称
        if random.random() >= 0.5:
            label = -label
            img_ori = cv2.flip(img_ori, 1)

        # process pts
        height, width = img_ori.shape[0:2]
        pts_ori[:,0] = pts_ori[:,0] * 1.0 / width
        pts_ori[:,1] = pts_ori[:,1] * 1.0 / height

        # process img,resize
        try:
            resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        except:
            print(self.imgPathList[index])
            sys.exit('some wrong in dataset')


        resized_im = resized_im * 0.0039216

        resized_im = resized_im[:,:,np.newaxis]

        resized_im = resized_im.transpose(2,0,1)


        eyePts = pts_ori[0:11,:].reshape((-1))
        mouthPts= pts_ori[11:,:].reshape((-1))

        return resized_im, torch.FloatTensor([label])

class DatasetWrinkle(torchDataset):

    def __init__(self, imgDir=[], size=128, imgChannel=1, isTrain='train'):

        self.size = size

        # get imgpath
        self.fileList = []
        for i in range(len(imgDir)):
            self.fileList += getlist(imgDir[i],'.jpg')
        random.shuffle(self.fileList)
        # print(len(self.fileList))
        self.imgPathList = []
        self.labelPathList = []

        self.fileList = list(filter(filterWrinkleImg,self.fileList))
        # print(len(self.fileList))
        # time.sleep(1000)
        # get labelpath
        for imgPath in self.fileList:

            xmlPath = imgPath.replace('.jpg','.xml')
            if os.path.exists(xmlPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(xmlPath)

        assert len(self.imgPathList) == len(self.labelPathList)

        labelList = [0] * 3
        for imgFile in self.imgPathList:
            # print(imgFile)
            dirname = osp.dirname(imgFile)
            # print(dirname)
            dirname = osp.basename(dirname)
            # print(dirname)
            label = int(dirname)
            labelList[label] += 1
            # import time
            # time.sleep(1000)

        # split train and validation
        nTrain = int(len(self.imgPathList) * 0.9)

        if isTrain == 'train':
            self.imgPathList = self.imgPathList[0:nTrain]
            self.labelPathList = self.labelPathList[0:nTrain]

        if isTrain == 'val':
            self.imgPathList = self.imgPathList[nTrain:]
            self.labelPathList = self.labelPathList[nTrain:]

        print('isTrain:',isTrain)
        print('labelList:',labelList)
        print('len(self.imgPathList):',len(self.imgPathList))
        print('len(self.labelPathList):',len(self.labelPathList))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPathList[index],0)
        box = getboxes(self.labelPathList[index])[0]
        box = [box]

        # get pos samples
        try:
            img_new, box_new = randomAug_box(img,box)
        except:
            print(self.imgPathList[index])
            print(box)
            sys.exit('img or label is wrong')

        height, width = img_new.shape[0:2]

        x1,y1,x2,y2 = box_new[0][0],box_new[0][1],box_new[0][2],box_new[0][3]
        w = x2 - x1
        h = y2 - y1

        assert w >= 50,' {} have wrong'.format(self.imgPathList[index])
        assert h >= 50,' {} have wrong'.format(self.imgPathList[index])

        if random.random() < 0.5:
            delta_x1 = np.random.randint(0,int(w * 0.25))
            delta_y1 = np.random.randint(0,int(h * 0.25))
            delta_x2 = np.random.randint(0,int(w * 0.25)) 
            delta_y2 = np.random.randint(0,int(h * 0.25)) 

        else:
            delta_x1 = np.random.randint(int(w * 0.25), int(w * 0.50))
            delta_y1 = np.random.randint(int(h * 0.25), int(h * 0.50))
            delta_x2 = np.random.randint(int(w * 0.25), int(w * 0.50)) 
            delta_y2 = np.random.randint(int(h * 0.25), int(h * 0.50)) 

        nx1 = max(x1 - delta_x1,0)
        ny1 = max(y1 - delta_y1,0)
        nx2 = min(x2 + delta_x2,width)
        ny2 = min(y2 + delta_y2,height)

        assert ny2 > ny1 + 50,' {} have wrong y1 {} y2 {} ny1 {} ny2 {} delta_y1 {} delta_y2 {}'.format(self.imgPathList[index],y1,y2,ny1,ny2,delta_y1,delta_y2)
        assert nx2 > nx1 + 50,' {} have wrong x1 {} x2 {} nx1 {} nx2 {} delta_x1 {} delta_x2 {}'.format(self.imgPathList[index],x1,x2,nx1,nx2,delta_x1,delta_x2)

        # 将点归一化到裁剪区域中
        x1 = (x1 - nx1) * 128 / (nx2 - nx1)
        y1 = (y1 - ny1) * 128 / (ny2 - ny1)
    
        x1 = x1 / 128.0000000000
        y1 = y1 / 128.0000000000

        x2 = (x2 - nx1) * 128 / (nx2 - nx1)
        y2 = (y2 - ny1) * 128 / (ny2 - ny1)
    
        x2 = x2 / 128.0000000000
        y2 = y2 / 128.0000000000

        cropped_im = img_new[int(ny1): int(ny2), int(nx1): int(nx2)]
        resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR).astype('float')*0.0039216

        resized_im = resized_im[np.newaxis,]

        dirname = osp.dirname(self.imgPathList[index])
        dirname = osp.basename(dirname)

        try:
            label = int(dirname)
        except:
            print(self.imgPathList[index])
            sys.exit()

        return resized_im,torch.FloatTensor([label]).type(torch.FloatTensor)
