'''
@Author: Jiangtao
@Date: 2019-09-30 16:32:57
@LastEditors: Jiangtao
@LastEditTime: 2019-10-18 17:29:55
@Description: 
'''
import caffe
import h5py
import cv2
import time
import math
import numpy as np 
import random
import os
import glob

def get_eye_pts_from_68_11(line):

    if len(line) == 6:
        eyePts = np.zeros((22,))
    elif len(line) == 5:
        eyePts = np.zeros((22,))
    else:
        pts = line[2:]
        pts = np.array(pts).reshape((68,1,2))
        eyePts = np.concatenate((pts[0,:],pts[36,:],pts[37,:],pts[39,:],pts[41,:],pts[27,:],pts[42,:],pts[44,:],pts[45,:],pts[46,:],pts[16,:]),axis=0)
    
    eyePts = eyePts.reshape((22,))

    return eyePts
    
def get_mouth_pts_from_68_8(line):
    
    if len(line) == 6:
        mouthPts = np.zeros((16,))
    elif len(line) == 5:
        mouthPts = np.zeros((16,))

    else:
        pts = line[2:]
        pts = np.array(pts).reshape((68,1,2))
        mouthPts = np.concatenate((pts[8,:],pts[30,:],pts[48,:],pts[51,:],pts[54,:],pts[57,:],pts[62,:],pts[66,:],),axis=0)

    mouthPts = mouthPts.reshape((16,))

    return mouthPts

def getTwoList(txtList):

    imgList = []
    pointList = []
    
    for txt in txtList:

        with open(txt) as f:
            lines = f.readlines()

        for line in lines:

            annotations = line.strip().split(' ')

            path = os.path.join('/home/jiangtao/80.50/jiangtao/',annotations[0]).replace('/./','/').replace('\\','/')
            eyePts = get_eye_pts_from_68_11(annotations)

            imgList.append(path)
            pointList.append(eyePts)

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(imgList)
    random.seed(randnum)
    random.shuffle(pointList)

    return imgList,pointList

def genH5(imgList,pointList,num):

    output = 'pos'

    for i in range(10):

        with h5py.File('/media/jiangtao/b087af92-91c9-4565-932d-0e0348bee2ca/dataset/trainset/results_eye/{}_'.format(num) + output + '_' + str(i) +'.h5', 'w') as h5:

            imgList_sp = imgList[int(i * 0.1 * len(imgList)):int((i+1) * 0.1 * len(imgList))]
            pointList_sp = pointList[int(i * 0.1 * len(imgList)):int((i+1) * 0.1 * len(imgList))]

            imgList_real = []

            for path in imgList_sp:
                
                img = cv2.imread(path,0)
                img = img.reshape((1,128,128))*0.0039216
                imgList_real.append(img)
                
            imgArray = np.array(imgList_real)
            pointArray = np.array(pointList_sp)

            print(imgArray.shape)
            print(pointArray.shape)
            
            h5['data'] = imgArray.astype(np.float32)
            h5['labels'] = pointArray.astype(np.float32)

            print('done')


if __name__ == '__main__':

	# imgList, pointList = getTwoList(['/home/jiangtao/80.50/jiangtao/128_new/pos_128_0_1.txt',
    #                                  '/home/jiangtao/80.50/jiangtao/128_new/mask_128_2_3.txt',
    #                                  '/home/jiangtao/80.50/jiangtao/128_new/hongkong_add_align.txt'])

	# print('geted list')
	# print(len(imgList))

	# for i in range(10):
	#    print('splited')

	#    imgList_ = imgList[int(i*(0.1)*len(imgList)):int((i+1)*(0.1)*len(imgList))]
	#    pointList_= pointList[int(i*(0.1)*len(imgList)):int((i+1)*(0.1)*len(imgList))]

	#    genH5(imgList_,pointList_,i)


	fileList = glob.glob('/media/jiangtao/b087af92-91c9-4565-932d-0e0348bee2ca/dataset/trainset/results_eye/*.h5')

	with open('/media/jiangtao/b087af92-91c9-4565-932d-0e0348bee2ca/dataset/trainset/results_eye/list_train.txt','w') as f:

	     fileList_ = fileList[0:-1]
	     print(fileList_)
	     for file in fileList_:
	         f.write(os.path.join('/',file))
	         f.write('\n')

	with open('/media/jiangtao/b087af92-91c9-4565-932d-0e0348bee2ca/dataset/trainset/results_eye/list_test.txt','w') as f:

	     fileList_ = fileList[-1:]
	     print(fileList_)
	     for file in fileList_:
	         f.write(os.path.join('/',file))
	         f.write('\n')

