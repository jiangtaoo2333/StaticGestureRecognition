'''
@Author: Jiangtao
@Date: 2019-07-10 09:48:59
* @LastEditors  : Please set LastEditors
* @LastEditTime : 2021-10-28 10:41:44
@Description: 
'''
import copy
import glob
from operator import index
import os
import os.path as osp
import random
import re
import shutil
import sys
import time
import uuid
import xml.dom.minidom
import xml.etree.ElementTree as ET
from multiprocessing import Process
from os import listdir
from xml.dom.minidom import Document

import cv2
import numpy as np
import skimage
import skimage.io as skio
import torch
from PIL import Image
from tqdm import tqdm
from xpinyin import Pinyin
import numpy as np
from sklearn.decomposition import PCA

def randomFlip(img,pts):

    height, width = img.shape[0:2]
    flipped_img = cv2.flip(img,1)
    pts_ = np.zeros_like(pts)
    pts_[:,1] = pts[:,1]
    pts_[:,0] = width - pts[:,0]
    pts = pts_

    nPoints = pts.shape[0]

    if 19 == nPoints:
        for i,j in [[0,10],[1,8],[2,7],[3,6],[4,9],[13,15]]:
            temp = copy.deepcopy(pts[i])
            pts[i] = pts[j]
            pts[j] = temp
    if 21 == nPoints:
        for i,j in [[0,12],[1,9],[2,8],[3,7],[4,10],[5,11],[15,17]]:
            temp = copy.deepcopy(pts[i])
            pts[i] = pts[j]
            pts[j] = temp

    return flipped_img,pts

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

def getbox(xmlFile):

    # xmlFile = imgFile.replace('/img','/xml')
    xmlFile = os.path.splitext(xmlFile)[0] + '.xml'

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

def picPostfix():  # 相册后缀的集合
    postFix = set()
    postFix.update(['bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif',
                    'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'JPG', 'raw', 'jpeg'])
    return postFix

def getDiff(width, high, image):  # 将要裁剪成w*h的image照片 
    diff = []
    im = image.resize((width, high))
    imgray = im.convert('L')  # 转换为灰度图片 便于处理
    pixels = list(imgray.getdata())  # 得到像素数据 灰度0-255

    for row in range(high): # 逐一与它左边的像素点进行比较
        rowStart = row * width  # 起始位置行号
        for index in range(width - 1):
            leftIndex = rowStart + index  
            rightIndex = leftIndex + 1  # 左右位置号
            diff.append(pixels[leftIndex] > pixels[rightIndex])

    return diff  #  *得到差异值序列 这里可以转换为hash码*

def getHamming(diff=[], diff2=[]):  # 暴力计算两点间汉明距离
    hamming_distance = 0
    for i in range(len(diff)):
        if diff[i] != diff2[i]:
            hamming_distance += 1

    return hamming_distance

def getALLext(dir):

    labelList = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if ext not in labelList:
                labelList.append(ext)
    return labelList

def png2jpg(dir):

    labelList = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if ext in ['.png','.JPG','.jpeg','.Jpeg','.JPEG','.PNG','.bmp','PNG']:
                labelList.append(os.path.join(root,name))
    n = 0
    N = len(labelList)
    for imgfile in labelList:

        n += 1
        print('{} in {}'.format(n,N))

        img = cv2.imread(imgfile,-1)

        filename = os.path.splitext(imgfile)[0] + '.jpg'
        cv2.imwrite(filename,img)

        os.remove(imgfile)

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # box = (x1, y1, x2, y2)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # abtain the offset of the interception of union between crop_box and gt_box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)

    return ovr

def getlist(dir,extension,Random=True):
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

def getDirs(dir):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            list.append(os.path.join(root, name))
    return list

def getDir(dir):
    list = []
    for root, dirs, files in os.walk(dir, topdown=True):
        for name in dirs:
            list.append(os.path.join(root,name))

    return list

def deletByExt(dir,extension):

    list = getlist(dir,extension)
    for file in tqdm(list):
        os.remove(file)

def deleteredundance(dir1,targetext1,dir2,targetext2):

    labelList = []
    for root, dirs, files in os.walk(dir2, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if targetext2 == ext:
                labelList.append(filename)

    print(len(labelList))


    fileList = []
    for root, dirs, files in os.walk(dir1, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if targetext1 == ext:
                fileList.append(os.path.join(root,name))

    print(len(fileList))

    for file in tqdm(fileList):

        file = file.replace('\\','/')
        filename = file.split('/')[-1]
        filename,_ = os.path.splitext(filename)

        if not filename in labelList:
            print(file)
            os.remove(file)

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def getchinesefile(dir,targetext):

    fileList = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename,ext = os.path.splitext(name)
            if targetext == ext:
                fileList.append(os.path.join(root,name))
    print(len(fileList))
    for file in fileList:
        if is_chinese(file):
            print(file)

def splitTrainTest(imageInput,imageOutput,ext1,labelInput,labelOutput,ext2,scale):

    imagelist = getlist(imageInput,ext1)
    random.shuffle(imagelist)

    if scale > 0:
        numTest = scale
    else:
        numTest = int(len(imagelist) * scale)

    imagelistTest = imagelist[0:numTest]

    for imageFile in imagelistTest:

        filename = os.path.splitext(os.path.basename(imageFile))[0]
        labelFile = labelInput + filename + ext2

        shutil.move(imageFile,imageOutput)
        shutil.move(labelFile,labelOutput)

def deleteerror(dir,ext):

    List = getlist(dir,ext)
    print(len(List))
    N = len(List)
    n = 0
    
    for file in List:
        n += 1
        print('{} in {}'.format(n,N))
        img = cv2.imread(file)
        if np.ndarray != type(img):
            print(file)
            os.remove(file)

def deleteSpace(dir,ext):

    fileList = getlist(dir,ext)
    print(len(fileList))

    num = 'jt0612'
    for File in fileList:
        
        if ' ' in File:
            print('{} have space'.format(File))
            if os.path.exists(File):
                os.rename(File,File.replace(' ',str(num)))

def bgr2gray(dir,ext):

    fileList = getlist(dir,ext)

    for file in fileList:

        img = cv2.imread(file,-1)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        cv2.imwrite(file,img)

def gray2bgr(dir,ext):

    fileList = getlist(dir,ext)

    for file in fileList:

        img = cv2.imread(file,-1)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        cv2.imwrite(file,img)

def insert(pts_origin,pts_insert):

    for i,j in [[0,0],[36,1],[37,2],[39,3],[41,4],[27,5],[42,6],[44,7],[45,8],[46,9],[16,10],
                [8,11],[30,12],[48,13],[51,14],[54,15],[57,16],[62,17],[66,18]]:

        pts_origin[i] = pts_insert[j]

    return pts_origin

def copydirbyext(dir,ext,dir2):

    fileList = getlist(dir,ext)
    for file in tqdm(fileList):

        try:
            shutil.copy(file,dir2)
        except:
            print('a')
            continue

def movedirbyext(dir,ext,dir2):

    fileList = getlist(dir,ext)


    for file in tqdm(fileList):

        try:
            shutil.move(file,dir2)
        except:
            print('move failed')
            continue

def deletDirs(target):
    
    dirs = os.listdir(target)

    for dir in dirs:

        dir = os.path.join(target,dir)
        if os.path.isdir(dir):
            shutil.rmtree(dir)

def unique(dir,diffnum):

    width = 32
    high = 32  # 压缩后的大小
    dirName = dir  # 相册路径
    allDiff = []
    postFix = picPostfix()  #  图片后缀的集合

    dirList = os.listdir(dirName)
    nBefore = len(dirList)
    cnt = 0
    for i in dirList:
        cnt += 1
        # print(cnt)  # 可以不打印 表示处理的文件计数
        if str(i).split('.')[-1] in postFix:  # 判断后缀是不是照片格式
            
            im = Image.open(os.path.join(dir,i))
            diff = getDiff(width, high, im)
            allDiff.append((str(i), diff))
            
    for i in range(len(allDiff)):
        for j in range(i + 1, len(allDiff)):
            if i != j:
                ans = getHamming(allDiff[i][1], allDiff[j][1])
                # print(ans)
                if ans <= diffnum:  # 判别的汉明距离，自己根据实际情况设置
                    # print(allDiff[i][0], "and", allDiff[j][0], "maybe same photo...")
                    try:
                        os.remove(os.path.join(dir,allDiff[i][0]))
                    except:
                        break
    
    dirList = os.listdir(dirName)
    nAfter = len(dirList)
    print('befor:{},after:{} '.format(nBefore,nAfter))

def splitby1000(dir,num=200):

    fileList = getlist(dir,'.jpg')
    N = int(len(fileList) / num)

    for i in range(N+1):

        dirpath = os.path.join(dir,str(i))

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        fileList_ = fileList[num*i:num*(i+1)]
        for fileName in fileList_:
            shutil.move(fileName,dirpath)
            try:
                shutil.move(fileName.replace('.jpg','.xml'),dirpath)
            except:
                continue

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    return (x, y, w, h)

def enlargeRoi(minx,miny,maxx,maxy,w,h):

    roiw = maxx - minx
    roih = maxy - miny

    minx -= roiw*0.35
    miny -= roih*0.35

    maxx += roiw*0.35
    maxy += roih*0.35

    minx = max(0,minx)
    miny = max(0,miny)

    maxx = min(w,maxx)
    maxy = min(h,maxy)

    return int(minx),int(miny),int(maxx),int(maxy)

def enlargeRoi_eye(minx,miny,maxx,maxy,w,h):

    roiw = maxx - minx
    roih = maxy - miny

    minx -= 20
    miny -= 30

    maxx += 20
    maxy += 30

    minx = max(0,minx)
    miny = max(0,miny)

    maxx = min(w,maxx)
    maxy = min(h,maxy)

    return int(minx),int(miny),int(maxx),int(maxy)

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    softmax = softmax.sum(axis=-2)
    return softmax

def reduceRedundance(dir1,ext1,dir2,ext2):
    deleteredundance(dir2,ext2,
                    dir1,ext1)
    deleteredundance(dir1,ext1,
                    dir2,ext2)

def deleteChinese(dir,ext):
    fileList = getlist(dir,ext)
    for file in fileList:
        if is_contains_chinese(file):
            os.remove(file)

def mergeTxtFile():

    txtList = getlist('/mnt/workspace/jiangtao/128_new/alignment/','.txt')
    
    for txtFile in txtList:

        dir = os.path.dirname(txtFile)
        filename = os.path.basename(txtFile)
        filename = os.path.splitext(filename)[0]

        if filename[-1] != '4':
            continue 

        filename = filename[:-2]
        
        with open('/mnt/workspace/jiangtao/128_new/alignment/{}.txt'.format(filename),'w') as f:
            for i in range(5):
                with open('/mnt/workspace/jiangtao/128_new/alignment/{}_{}.txt'.format(filename,i),'r') as f1:
                    lines = f1.readlines()
                for line in lines:
                    line = line.strip()

                    if(len(line.split(' ')) != 138):
                        continue
                    
                    f.write(line)
                    f.write('\n')

def mastOnImage(img,x1,y1,x2,y2):

    assert len(img.shape) == 2,'img dims is wrong'
    assert type(x1) is int and type(x1) is int,'location is not int'

    resized_im = np.random.randint(255,size=(y2-y1,x2-x1))
    img[y1:y2,x1:x2] = resized_im
    return img

def createMaskImage():

    imgList = getlist('/home/jiangtao/workspace/dataset/test/detectTest/','.jpg')


    for imgFile in tqdm(imgList):

        xmlFile = imgFile.replace('.jpg','.xml')

        img = cv2.imread(imgFile,0)
        box = getbox(xmlFile)

        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[0][2])
        y2 = int(box[0][3])
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        label = 1.0
        if(random.random() > 0.65):
            label -= 0.25
            img = mastOnImage(img,x1,y1,x_center,y_center)

        if(random.random() > 0.65):
            label -= 0.25
            img = mastOnImage(img,x_center,y1,x2,y_center)

        if(random.random() > 0.65):
            label -= 0.25
            img = mastOnImage(img,x1,y_center,x_center,y2)

        if(random.random() > 0.65):
            label -= 0.25
            img = mastOnImage(img,x_center,y_center,x2,y2)

        label = round(label,2)
        imgFile = os.path.join('/home/jiangtao/workspace/dataset/test/FaceAreaTest/{}'.format(label),os.path.basename(imgFile))
        cv2.imwrite(imgFile,img)
        try:
            shutil.copy(xmlFile,'/home/jiangtao/workspace/dataset/test/FaceAreaTest/{}'.format(label))
        except:
            print('already have')

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

    assert pts.shape == (21,2) or pts.shape == (19,2)

    return pts

def checkXml(SrcDir):

    xmlList = getlist(SrcDir,'.xml')
    for xmlFile in tqdm(xmlList):

        box = getbox(xmlFile)
        minx = box[0][0]
        miny = box[0][1]
        maxx = box[0][2]
        maxy = box[0][3]

        if maxx <= (minx+10) or maxy<=(miny+10):
            # os.remove(xmlFile)
            print(xmlFile)

def xml2txt(SrcDir):

    xmlList = getlist(SrcDir,'.xml')
    
    for xmlFile in tqdm(xmlList):

        dom = xml.dom.minidom.parse(xmlFile)  
        root = dom.documentElement

        itemlist = root.getElementsByTagName('height')
        height = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('width')
        width = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('xmin')
        minX = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('ymin')
        minY = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('xmax')
        maxX = int(float(itemlist[0].firstChild.data))

        itemlist = root.getElementsByTagName('ymax')
        maxY = int(float(itemlist[0].firstChild.data))

        W = maxX - minX
        H = maxY - minY

        centerX = (minX + maxX) / 2
        centerY = (minY + maxY) / 2

        centerX /= float(width)
        centerY /= float(height)
        W /= float(width)
        H /= float(height)

        filname,ext = os.path.splitext(xmlFile)
        boxFile = filname + '.txt'

        with open(boxFile,'w') as f:
            f.write('0')
            f.write(' ')
            f.write(str(float(centerX)))
            f.write(' ')
            f.write(str(float(centerY)))
            f.write(' ')
            f.write(str(float(W)))
            f.write(' ')
            f.write(str(float(H)))

def deleteDirbyExt(dir,ext):

    fileList = getlist(dir,ext)
    for afile in tqdm(fileList):
        os.remove(afile)

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

def findLarge():

    dir1 = '/jiangtao2/dataset/train/alignment/'
    print(dir1)
    imgList = getlist(dir1,'.jpg')

    rightRes = [0] * 10
    leftRes = [0] * 10

    for imgFile in tqdm(imgList):

        imgFile = imgFile.replace('\\','/')
        filename = os.path.splitext(imgFile)[0]
        ptsFile = filename + '.pts'

        if not os.path.exists(ptsFile):
            print(imgFile)
            continue

        pts = getPts(ptsFile)[0]
        npoints = getnpoints(ptsFile)[0]

        if(21 == npoints):
            mask = 'unmask'
        else:
            mask = 'mask'

        left = pts[0,0]
        mid = pts[6,0]
        right = pts[12,0]


        scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
        scale2 = abs((right-mid) / ((mid-left)+1e-5))

        scale = max(scale1,scale2)
        
        
        if scale1 >= 1:
            index = (int(scale1/1))
            if(index >= 9):
                index = 9
            rightRes[index]+=1

            targetDir = '/jiangtao2/dataset/train/angle/DSM/{}/rignt/{}'.format(mask,str(index))

            if not os.path.exists(targetDir):
                os.makedirs(targetDir)

            try:
                shutil.copy(imgFile,targetDir)
                shutil.copy(ptsFile,targetDir)
            except:
                continue


        elif scale2 >= 1:
            index = (int(scale2/1))
            if(index >= 9):
                index = 9
            leftRes[index]+=1

            targetDir = '/jiangtao2/dataset/train/angle/DSM/{}/left/{}'.format(mask,str(index))

            if not os.path.exists(targetDir):
                os.makedirs(targetDir)

            try:
                shutil.copy(imgFile,targetDir)
                shutil.copy(ptsFile,targetDir)
            except:
                continue


        else:
            continue

    print(rightRes)
    print(leftRes)

def splitLarge():

    # 获取所有图片
    imgList = getlist('/home/jiangtao/share/train/alignment/mask/img_Angle/','.jpg')

    middle = 0
    small_left = 0
    large_left = 0
    small_right = 0
    large_right = 0

    for imgFile in tqdm(imgList):

        ptsFile = imgFile.replace('.jpg','.pts')
        if not os.path.exists(ptsFile):
            print(imgFile)
            os.remove(imgFile)
            continue
        
        # 获取关键点
        try:
            pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1).reshape((-1,2))
        except:
            print(imgFile)
            os.remove(imgFile)
            os.remove(ptsFile)
            continue
        
        if 21 == pts.shape[0]:
            
            left = pts[0,0]
            mid = pts[6,0]
            right = pts[12,0]

        elif 19 == pts.shape[0]:
            left = pts[0,0]
            mid = pts[5,0]
            right = pts[10,0]
        
        else:
            print(imgFile)
            os.remove(imgFile)
            os.remove(ptsFile)
            continue
        
        # 统计左右比例
        scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
        scale2 = abs((right-mid) / ((mid-left)+1e-5))

        # if max(scale1,scale2) <= 3:
        #     middle += 1

        # 根据左右比例进行划分
        if scale1 > scale2:

            # if(3<scale1<=8):
            #     small_right += 1

            # if(8<scale1):
            #     large_right += 1
            #     try:
            #         shutil.copy(imgFile,'/home/jiangtao/share/train/alignment/unmask/img_Angle/')
            #         shutil.copy(ptsFile,'/home/jiangtao/share/train/alignment/unmask/img_Angle/')
            #     except:
            #         continue

            shutil.move(imgFile,'/home/jiangtao/share/train/alignment/mask/img_Angle/right/')
            shutil.move(ptsFile,'/home/jiangtao/share/train/alignment/mask/img_Angle/right/')

        if scale2 > scale1:

            # if(3<scale2<=8):
            #     small_left += 1
            # if(8<scale2):
            #     large_left += 1
            #     try:
            #         shutil.copy(imgFile,'/home/jiangtao/share/train/alignment/unmask/img_Angle/')
            #         shutil.copy(ptsFile,'/home/jiangtao/share/train/alignment/unmask/img_Angle/')
            #     except:
            #         continue

            shutil.move(imgFile,'/home/jiangtao/share/train/alignment/mask/img_Angle/left/')
            shutil.move(ptsFile,'/home/jiangtao/share/train/alignment/mask/img_Angle/left/')

    # print(middle)
    # print(small_left)
    # print(large_left)
    # print(small_right)
    # print(large_right)

def flipImage():

    imgList = getlist('/home/jiangtao/workspace/dataset/train/angle/DSM/mask/right/','.jpg')

    for imgFile in tqdm(imgList):
        img = cv2.imread(imgFile,1)
        img = cv2.flip(img,1)
        cv2.imwrite(imgFile.replace('/right/','/left_/'),img)

    imgList = getlist('/home/jiangtao/workspace/dataset/train/angle/DSM/mask/left/','.jpg')

    for imgFile in tqdm(imgList):
        img = cv2.imread(imgFile,1)
        img = cv2.flip(img,1)
        cv2.imwrite(imgFile.replace('/left/','/right_/'),img)

# 随机HSV空间变化
def randomHSV(img):

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

    return img

# 随机滤波
def randomBlur(img):

    n = random.random()
    if n<=0.25:
        img_mean = cv2.blur(img, (5,5))
        return img_mean
    if n <= 0.5:
        img_Guassian = cv2.GaussianBlur(img,(5,5),0)
        return img_Guassian
    if n <= 0.75:
        img_median = cv2.medianBlur(img, 5)
        return img_median
    if n <= 1:
        img_bilater = cv2.bilateralFilter(img,9,75,75)

        return img_bilater

# 随机噪声
def randomNoise(img):

    N = int(img.shape[0] * img.shape[1] * 0.001)
    for i in range(N): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255

    return img

def enlargeDataset(imgList,dirT):
    
    for i in range(50):
        print(i)
        for imgFile in imgList:

            img = cv2.imread(imgFile,1)
            
            if random.random() > 0.5:
                img = randomHSV(img)

            if random.random() > 0.5:
                img = randomBlur(img)

            # if random.random() > 0.8:
            #     img = randomNoise(img)

            basename = os.path.basename(imgFile).replace('.jpg','_{}.jpg'.format(i))
            imgFileNew = dirT+basename
            xmlFileNew = imgFileNew.replace('.jpg','.pts')

            cv2.imwrite(imgFileNew,img)
            shutil.copy(imgFile.replace('.jpg','.pts'),xmlFileNew)

def chinesetopinyin(imgList):

    p = Pinyin()
    for imgFile in tqdm(imgList):
        # print(imgFile)
        if(imgFile != p.get_pinyin(imgFile)):
            print(imgFile)
            os.rename(imgFile,p.get_pinyin(imgFile))

def removeDemageImg():

    imgList = getlist('/home/jiangtao/workspace/dataset/train/cover/add/BackLighting/','.jpg')
    for imgFile in tqdm(imgList):
        img = cv2.imread(imgFile,1)
        if(type(img)!=type(np.zeros(5))):
            print(imgFile)
            print(type(img))

def calculateMeanAndStd(imgList):

    R_channel = 0
    G_channel = 0
    B_channel = 0
    R_channel_square = 0
    G_channel_square = 0
    B_channel_square = 0
    pixels_num = 0
    
    imgs = []
    for imgFile in tqdm(imgList):
        img = cv2.imread(imgFile,1)
        h, w, _ = img.shape
        pixels_num += h*w       # 统计单个通道的像素数量
    
        R_temp = img[:, :, 0]
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))
        G_temp = img[:, :, 1]
        G_channel += np.sum(G_temp)
        G_channel_square += np.sum(np.power(G_temp, 2.0))
        B_temp = img[:, :, 2]
        B_channel = B_channel + np.sum(B_temp)
        B_channel_square += np.sum(np.power(B_temp, 2.0))
    
    R_mean = R_channel / pixels_num
    G_mean = G_channel / pixels_num
    B_mean = B_channel / pixels_num
    
    """  
    S^2
    = sum((x-x')^2 )/N = sum(x^2+x'^2-2xx')/N
    = {sum(x^2) + sum(x'^2) - 2x'*sum(x) }/N
    = {sum(x^2) + N*(x'^2) - 2x'*(N*x') }/N
    = {sum(x^2) - N*(x'^2) }/N
    = sum(x^2)/N - x'^2
    """
    
    R_std = np.sqrt(R_channel_square/pixels_num - R_mean*R_mean)
    G_std = np.sqrt(G_channel_square/pixels_num - G_mean*G_mean)
    B_std = np.sqrt(B_channel_square/pixels_num - B_mean*B_mean)
    
    return B_mean,G_mean,R_mean,B_std,G_std,R_std

def splitTrainTestVoc():

    imgList = getlist('/home/jiangtao/workspace/dataset/train/detect_voc/JPEGImages/','.jpg')
    random.shuffle(imgList)
    print(len(imgList))

    with open('/home/jiangtao/workspace/dataset/train/detect_voc/ImageSets/Main/trainval.txt','w') as f:
        for imgFile in imgList[0:int(0.8*len(imgList))]:
            f.write(os.path.splitext(os.path.basename(imgFile))[0])
            f.write('\n')
            
    with open('/home/jiangtao/workspace/dataset/train/detect_voc/ImageSets/Main/test.txt','w') as f:
        for imgFile in imgList[int(0.8*len(imgList)):]:
            f.write(os.path.splitext(os.path.basename(imgFile))[0])
            f.write('\n')

def flipImgAndPts():

    imgList = getlist('/jiangtao2/dataset/train/alignment/unmask/img_Angle/','.jpg')

    for imgFile in tqdm(imgList):
        imgFile = imgFile.replace('\\','/')
        img = cv2.imread(imgFile,-1)
        ptsFile = imgFile.replace('.jpg','.pts')
        pts = getpoints(ptsFile)

        nPoints = pts.shape[0]

        if 19 == nPoints:
            flipImg,flipPts = randomFlip(img,pts)
            
            if ('/left/' in imgFile):
                imgFile = imgFile.replace('/left/','/right/')
                ptsFile = ptsFile.replace('/left/','/right/')
            else:
                imgFile = imgFile.replace('/right/','/left/')
                ptsFile = ptsFile.replace('/right/','/left/')

            cv2.imwrite(imgFile.replace('/img_Angle/','/img_Angle_flip/'),flipImg)
            with open(ptsFile.replace('/img_Angle/','/img_Angle_flip/'),'w') as f:
                f.write('version: 1')
                f.write('\n')
                f.write('npoints: 19')
                f.write('\n')
                f.write('{')
                f.write('\n')

                for i in range(19):
                    f.write(str(flipPts[i][0]))
                    f.write(' ')
                    f.write(str(flipPts[i][1]))
                    f.write('\n')

                f.write('}')

        if 21 == nPoints:

            flipImg,flipPts = randomFlip(img,pts)
            cv2.imwrite(imgFile.replace('/img_Angle/','/img_Angle_flip/'),flipImg)
            
            with open(ptsFile.replace('/img_Angle/','/img_Angle_flip/'),'w') as f:
                f.write('version: 1')
                f.write('\n')
                f.write('npoints: 21')
                f.write('\n')
                f.write('{')
                f.write('\n')

                for i in range(21):
                    f.write(str(flipPts[i][0]))
                    f.write(' ')
                    f.write(str(flipPts[i][1]))
                    f.write('\n')

                f.write('}')

def renameDirs(target):
    
    dirs = os.listdir(target)

    i = 0
    for dir in dirs:
        dir = os.path.join(target,dir)
        if os.path.isdir(dir):
            os.rename(dir,os.path.dirname(dir) + '/__{}/'.format(i))
        i += 1

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

    elif 21 == pts.shape[0]:
        pass

    return pts

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

def checkPts(dir):

    imgList = getlist(dir,'.jpg')

    for imgFile in tqdm(imgList):

        # print(imgFile)
        ptsFile = imgFile.replace('.jpg','.pts')

        try:
            img = cv2.imread(imgFile,0)
        except:
            print('{} can not open'.format(imgFile))
            os.remove(imgFile)
            continue

        try:
            height,width = img.shape[0:2]
        except:
            print('{} can not get w and h'.format(imgFile))
            os.remove(imgFile)
            continue

        if not os.path.exists(ptsFile):
            print('dont have {}'.format(ptsFile))
            # os.remove(imgFile)
            continue
        
        try:
            pts = getpoints(ptsFile)
        except:
            print('cannot open {}'.format(ptsFile))
            # os.remove(imgFile)
            continue


        minX,minY = pts.min(axis=0)
        maxX,maxY = pts.max(axis=0)

        if maxX >= (width+5) or maxY >= (height+5) or minX < 0 or minY < 0:
            print('{} cross the border'.format(imgFile))
            continue

    reduceRedundance(dir,'.jpg',dir,'.pts')

def isValidXml(xmlFile):

    if not os.path.exists(xmlFile):
        return False

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

def checkXml(dir):

    imgList = getlist(dir,'.jpg')

    f = open('./errorXml.txt','w')
    for imgFile in tqdm(imgList):

        xmlFile = imgFile.replace('.jpg','.xml')
        if(not isValidXml(xmlFile)):
            try:
                f.write(xmlFile + '\n')
                f.write(imgFile + '\n')
            except:
                continue
    f.close()
    # reduceRedundance(dir,'.jpg',dir,'.xml')

def calculateAngle(dir):

    # 获取所有图片
    ptsList = getlist(dir,'.txt')

    leftList = []
    rightList = []
    middle = 0

    for ptsFile in tqdm(ptsList):

        # 获取关键点
        try:
            pts = np.genfromtxt(ptsFile,skip_header=1).reshape((-1,2))
        except:
            print(imgFile)
            continue
        
        if 21 == pts.shape[0]:
            left = pts[0,0]
            mid = pts[6,0]
            right = pts[12,0]

        elif 19 == pts.shape[0]:
            left = pts[0,0]
            mid = pts[5,0]
            right = pts[10,0]
        
        elif 106 == pts.shape[0]:
            left = pts[74,0]
            mid = pts[51,0]
            right = pts[85,0]
        
        # 统计左右比例
        scale1 = abs((mid-left) / ((right-mid)+1e-5)) 
        scale2 = abs((right-mid) / ((mid-left)+1e-5))

        if max(scale1,scale2) <= 3:
            middle += 1
            continue

        # 根据左右比例进行划分
        if scale1 > scale2:
            rightList.append(scale1)

        if scale2 > scale1:
            leftList.append(scale2)
    
    rightList = np.array(rightList)
    leftList = np.array(leftList)

    return rightList,leftList

def getMeanShape(ptsList):

    pointsArray = []
    for ptsFile in tqdm(ptsList):
        pts = np.genfromtxt(ptsFile,skip_header=1).reshape((-1,2))
        pointsArray.append(pts)

    pointsArray = np.array(pointsArray)

    return pointsArray.mean(axis=0)

#Procrustes analysis
def transformation_from_points(points1, points2):
    '''0 - 先确定是float数据类型 '''
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    '''1 - 消除平移的影响 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    '''2 - 消除缩放的影响 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    '''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T # 2xN
    B = points2.T # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)

    '''4 - 构建仿射变换矩阵 '''
    s = s2/s1
    sR = s*R
    c1 = c1.reshape(2,1)
    c2 = c2.reshape(2,1)
    T = c2 - np.dot(sR,c1) # 模板人脸的中心位置减去 需要对齐的中心位置（经过旋转和缩放之后）

    trans_mat = np.hstack([sR,T])   # 2x3

    return trans_mat

def getPCAshape(ptsList,meanShape):

    PCAArray = []
    for ptsFile in tqdm(ptsList):
        pts = np.genfromtxt(ptsFile,skip_header=1).reshape((-1,2))
        PCAArray.append(transformation_from_points(pts,meanShape))

    PCAArray = np.array(PCAArray)
    return PCAArray

def PCATo1Demision(pointsArray):
    pass

def writemeanFileProto():

    MEAN_NPY_PATH = 'mean.npy'
    
    mean = np.ones([3,224, 224], dtype=np.float)
    mean[0,:,:] = 123
    mean[1,:,:] = 123
    mean[2,:,:] = 123
    
    np.save(MEAN_NPY_PATH, mean)

    import caffe
    import sys

    mean_npy = np.load("mean.npy")
    blob = caffe.io.array_to_blobproto(mean_npy)
    mean_binproto = "mean.binaryproto"
    with open(mean_binproto, 'wb') as f:
        f.write(blob.SerialToString())

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

def writeNewPts(targetdir):

    ptsList = getlist(targetdir,'.pts')

    for ptsFile in tqdm(ptsList):

        numpoints = 0
        with open(ptsFile,'r') as f:
            lines = f.readlines()

            if('version: new' in lines[0].strip()):
                continue

            numLines = len(lines)

            if(23 == numLines):
                numpoints = 19
            elif(25 == numLines):
                numpoints = 21
            else:
                print('some thing is wrong {}'.format(ptsFile))
                break

        pts = np.genfromtxt(ptsFile,skip_header=3,skip_footer=1)
        if(19 == numpoints):
            pts = pts19to21(pts)
            numpoints = 21

        assert 21 == numpoints,'not 21 points'
        if('mask' in ptsFile):
            numpoints = 13
        with open(ptsFile.replace('.pts','.pts'),'w') as f:

            f.write('version: new')
            f.write('\n')

            f.write('npoints: {}'.format(numpoints))
            f.write('\n')
            f.write('{')
            f.write('\n')
            for i in range(pts.shape[0]):
                f.write('{} {}'.format(pts[i][0],pts[i][1]))
                f.write('\n')
            f.write('}')
            # f.write('\n')

def splitgazeDMS():

    imgList = getlist('/jiangtao2/dsm_gaze/','.jpg')


    for imgFile in tqdm(imgList):
        imgFile = imgFile.replace('\\','/')
        xmlFile = imgFile.replace('.jpg','.xml')
        ptsFile = imgFile.replace('.jpg','.pts')

        if not os.path.exists(xmlFile):
            continue
        if not os.path.exists(ptsFile):
            continue

        if('/eyeCentre/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeCenter')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeCenter')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeCenter')
            except:
                continue
        if('/eyeDown/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeDown')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeDown')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeDown')
            except:
                continue
        if('/up/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeUp')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeUp')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeUp')
            except:
                continue

        if('/eyeLeft/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeft')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeft')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeft')
            except:
                continue
        if('/eyeLeft_Down/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftDown')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftDown')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftDown')
            except:
                continue
        if('/eyeLeft_Up/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftUp')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftUp')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeLeftUp')
            except:
                continue

        if('/eyeRight/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeRight')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeRight')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeRight')
            except:
                continue
        if('/eyeRight_Down/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightDown')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightDown')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightDown')
            except:
                continue
        if('/eyeRight_Up/' in imgFile):
            try:
                shutil.copy(imgFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightUp')
                shutil.copy(xmlFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightUp')
                shutil.copy(ptsFile,'/jiangtao2/dataset/train/gazeDMS/eyeRightUp')
            except:
                continue

def splitgazeDMS_():
    srcDir = '/jiangtao2/dataset/train/gazeDMS/'

    dirList = os.listdir(srcDir)
    dirdict = {}
    for dirfile in dirList:
        realDir = os.path.join(srcDir,dirfile)
        reduceRedundance(realDir,'.jpg',realDir,'.xml')
        imgList = getlist(realDir,'.jpg')
        dirdict[dirfile] = len(imgList)
        nTrain = int(len(imgList) * 0.8)

        with open('/jiangtao2/dataset/train/gazeDMS/Train.txt','a') as f:
            for imgFile in imgList[0:nTrain]:
                f.write(imgFile)
                f.write('\n')

        with open('/jiangtao2/dataset/train/gazeDMS/Test.txt','a') as f:
            for imgFile in imgList[nTrain:]:
                f.write(imgFile)
                f.write('\n')
    print(dirdict)

def removeerrorSamples():

    with open('../../errorSamples.txt','r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        imgFile = line.strip()
        xmlFile = imgFile.replace('.jpg','.xml')

        try:
            os.remove(imgFile)
            os.remove(xmlFile)
        except:
            print(imgFile)
            continue

    with open('../../errorSamplesAlign.txt','r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        imgFile = line.strip()
        ptsFile = imgFile.replace('.jpg','.pts')

        try:
            os.remove(imgFile)
            os.remove(ptsFile)
        except:
            print(imgFile)
            continue

def getExtFile(imgFile,ext):

    filename = os.path.splitext(imgFile)[0]
    return filename + ext

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

def TimeStampToTime(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)

if __name__ == '__main__':


    overlaps = torch.randn(3,4)

    best_prior_overlap,best_prior_idx  = overlaps.max(1,keepdim=True)

    best_truth_overlap,best_truth_idx  = overlaps.max(0,keepdim=True)

    print(overlaps)
    print(best_prior_overlap)
    print(best_prior_idx)
    print(best_truth_overlap)
    print(best_truth_idx)
    best_prior_idx.squeeze_(1)       #（N）
    best_prior_overlap.squeeze_(1)   #（N）
    best_truth_idx.squeeze_(0)       #（M）
    best_truth_overlap.squeeze_(0)   #（M）

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    print(best_truth_overlap)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    print(best_truth_idx)








