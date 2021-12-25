import os
import random
import cv2
import copy

import numpy as np
import torch

from torch.utils.data.dataset import Dataset as torchDataset
import xml.etree.ElementTree as ET
from tqdm import tqdm


from .dataset_kl_occ import list_dirs, get_files, getlist
from .dataset_kl_occ import randomeResize_, randomFlip, randomeResizeFix, randomAug_box

def parse_xml(args, label_ids={'niguang':0, 'normal':1}):
    """parse xml data
    Ref link: https://github.com/HRNet/HRNet-Object-Detection/blob/master/tools/convert_datasets/pascal_voc.py
    :param args:
        xml_path & img_path.
    :return:
        parsed annotations like voc results of HRNet Detection.
    """
    img_path, xml_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    d = int(size.find('depth').text)
    clses = []
    bboxes = []
    labels = []

    for obj in root.findall('object'):
        # traditional laebls parsing
        name = obj.find('name').text
        clses.append(name)
        label = label_ids[name]
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]

        bboxes.append(bbox)
        labels.append(label)

    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0,))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
 
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'depth': d,
        'ann': {
            'clses': clses,
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
        }
    }

    assert len(bboxes) == 1

    return annotation['ann']['bboxes'], annotation['ann']['labels'][0]



class DatasetKLQualityCls(torchDataset):
    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train'):
        self.size = size
        self.channel = channel

        # get image path
        self.fileList = []
        if isinstance(imgDir, list):
            self.fileList.extend(get_files(imgDir, '.jpg'))
        else:
            self.fileList - get_files(imgDir, '.jpg')
        # NOTE & BUG: every train & test results is different.
        random.shuffle(self.fileList)

        self.imgPathList = []
        self.labelPathList = []
        self.is_train = True if isTrain == 'train' else False

        # get label path
        for imgPath in self.fileList:
            ptsPath = imgPath.replace('.jpg', '.xml')
            if os.path.exists(ptsPath):
                self.imgPathList.append(imgPath)
                self.labelPathList.append(ptsPath)

        print('=> img {} & label {}'.format(len(self.imgPathList), len(self.labelPathList)))
        assert len(self.imgPathList) == len(self.labelPathList)

        print('=> {} dataset load {} samples'.format(isTrain, len(self.imgPathList)))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        img = cv2.imdecode(np.fromfile(self.imgPathList[index], dtype=np.uint8), 0)

        # random aug img
        bbox, label = parse_xml((self.imgPathList[index], self.labelPathList[index]))
        img_ori, _, _ = randomAug_box(img=img, box=bbox, label=label, is_train=self.is_train)
        resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float') * 0.0039216

        cv2.imshow('img', img)
        cv2.imshow('resized_im', resized_im)
        print(label)
        if label > 0:
            info = 'niguang'
        else:
            info = 'normal'

        cv2.putText(img, info, (0, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        if cv2.waitKey(20000) & 0xFF == ord('q'):
            exit
            

        resized_im = resized_im[np.newaxis,]
        return resized_im, label, self.imgPathList[index]


if __name__ == '__main__':
    # check Premature end of JPEG file images
    imgDir = '/home/andrew/datasets/face/face_occlusion/train/train_jingxi_from_faces'
    fileList = get_files(imgDir, '.jpg')
    from skimage import io

    for item in tqdm(fileList):
        try:
            io.imread(item)
        except Exception as e:
            print(item)
