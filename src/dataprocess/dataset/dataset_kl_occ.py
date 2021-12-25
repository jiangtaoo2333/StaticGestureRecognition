import os
import random
import cv2
import copy

import numpy as np
import torch

from torch.utils.data.dataset import Dataset as torchDataset
import xml.etree.ElementTree as ET
from tqdm import tqdm


def list_dirs(path, recursive=True):
    out_files = []
    if recursive:
        for file_dir, folders, files in os.walk(path):
            for file in files:
                out_files.append(os.path.join(file_dir, file))
    else:
        for file in os.listdir(path):
            out_files.append(os.path.join(path, file))
    return out_files


def get_files(paths, targets='.xml', with_recurrence=True):
    out_files = []
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(targets, str):
        targets = [targets]
    for path in paths:
        if with_recurrence:
            files = list_dirs(path)
            for file in tqdm(files):
                if os.path.splitext(file)[1] in targets:
                    out_files.append(file)
        else:
            for file in tqdm(os.scandir(path)):
                if os.path.splitext(file)[1] in targets:
                    out_files.append(file.path)
    return out_files


def getlist(dir, extension):
    list = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            filename, ext = os.path.splitext(name)
            if extension == ext:
                list.append(os.path.join(root, name))

    # NOTE: correct read and show
    # print('=> {}'.format(len(list)))
    return list


def parse_xml(args):
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
    bboxes = []
    labels = []
    parts = []
    bboxes_ignore = []
    labels_ignore = []
    parts_ignore = []
    for obj in root.findall('object'):
        # traditional laebls parsing
        name = obj.find('name').text
        # label = label_ids[name]
        label = 1
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]

        # partial occlusion labels parsing
        eye_left = int(obj.find('eye_left').text)
        eye_right = int(obj.find('eye_right').text)
        cheek_left = int(obj.find('cheek_left').text)
        cheek_right = int(obj.find('cheek_right').text)
        nose = int(obj.find('nose').text)
        mouth = int(obj.find('mouth').text)
        jaw = int(obj.find('jaw').text)
        part = [eye_left, eye_right, cheek_left, cheek_right, nose, mouth, jaw]

        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
            parts_ignore.append(part)
        else:
            bboxes.append(bbox)
            labels.append(label)
            parts.append(part)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0,))
        parts = np.zeros((0, 7))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
        parts = np.array(parts)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))
        parts_ignore = np.zeros((0, 7))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
        parts_ignore = np.array(parts_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'depth': d,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64),
            'parts': parts.astype(np.int64),
            'parts_ignore': parts_ignore.astype(np.int64)
        }
    }

    return annotation['ann']['bboxes'], annotation['ann']['parts'][0]


def randomeResize_(img, box, scale=0.75):
    # scale just to prevent the moved area to be too large
    height, width = img.shape[0:2]
    minX, minY, maxX, maxY = box[0]
    w = (maxX - minX) * scale
    h = (maxY - minY) * scale
    dice = random.random()

    w_large_thr, h_large_thr = int(w * 0.75), int(h * 0.75)
    w_medium_thr, h_medium_thr = int(w * 0.375), int(h * 0.375)

    if dice < 0.5:
        delta_x1 = np.random.randint(w_medium_thr, w_large_thr)
        delta_y1 = np.random.randint(h_medium_thr, h_large_thr)
        delta_x2 = np.random.randint(w_medium_thr, w_large_thr)
        delta_y2 = np.random.randint(h_medium_thr, h_large_thr)
    else:
        delta_x1 = np.random.randint(0, w_medium_thr)
        delta_y1 = np.random.randint(0, h_medium_thr)
        delta_x2 = np.random.randint(0, w_medium_thr)
        delta_y2 = np.random.randint(0, h_medium_thr)

    # crop image
    nx1 = max(minX - delta_x1, 0)
    ny1 = max(minY - delta_y1, 0)
    nx2 = min(maxX + delta_x2, width)
    ny2 = min(maxY + delta_y2, height)

    # adjust bbox
    box[0][0] -= nx1
    box[0][2] -= nx1
    box[0][1] -= ny1
    box[0][3] -= ny1

    img = img[int(ny1): int(ny2), int(nx1): int(nx2)]
    return img, box


def randomFlip(img, box, label):
    """
    label index :       0           1       2           3           4     5     6
    label str:      eye_left, eye_right, cheek_left, cheek_right, nose, mouth, jaw
    """
    # flip image
    height, width = img.shape[0:2]
    flipped_img = cv2.flip(img, 1)

    # flip bbox
    minX, minY, maxX, maxY = box[0]
    box[0][0] = width - maxX
    box[0][2] = width - minX

    # flip label
    label_dst = copy.deepcopy(label)
    convert_idx = [1, 0, 3, 2, 4, 5, 6]
    for idx in range(len(convert_idx)):
        label_dst[idx] = label[convert_idx[idx]]

    return flipped_img, box, label_dst


def randomeResizeFix(img, box, is_train=True, scale=1.0):
    height, width = img.shape[0:2]
    minX, minY, maxX, maxY = box[0]

    # rescale bbox
    w = (maxX - minX) * scale
    h = (maxY - minY) * scale
    delta_x1, delta_x2 = w * 0.15, w * 0.15
    delta_y1, delta_y2 = h * 0.25, h * 0.1

    # bbox x1, y1 not changed
    nx1 = int(max(minX - delta_x1, 0))
    ny1 = int(max(minY - delta_y1, 0))
    nx2 = int(min(maxX + delta_x2, width))
    ny2 = int(min(maxY + delta_y2, height))

    img = img[ny1: ny2, nx1: nx2]

    box[0] = [nx1, ny1, nx2, ny2]
    return img, box


def randomAug_box(img, box, label=None, is_train=False):
    if (is_train):
        img, box = randomeResize_(img, box, scale=1.0)
        if random.random() > 0.5:
            img, box, label = randomFlip(img, box, label=label)
    else:
        # img, box = randomeResizeFix(img, box, scale=1.0)
        img, box = randomeResizeFix(img, box, scale=1.5)
    return img, box, label


class DatasetSpFaceDet(torchDataset):
    def __init__(self, imgDir=None, size=128, channel=1, isTrain='train'):
        self.size = size
        self.channel = channel

        # get image path
        self.fileList = get_files(imgDir, '.jpg')
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

        # split train and validation
        if isTrain != 'test':
            nTrain = int(len(self.imgPathList) * 0.9)
            if isTrain == 'train':
                # self.imgPathList = self.imgPathList[0:nTrain]
                # self.labelPathList = self.labelPathList[0:nTrain]
                print('use all dataset!!!')
                self.imgPathList = self.imgPathList
                self.labelPathList = self.labelPathList
            if isTrain == 'val':
                self.imgPathList = self.imgPathList[nTrain:]
                self.labelPathList = self.labelPathList[nTrain:]

        print('=> {} dataset load {} samples'.format(isTrain, len(self.imgPathList)))

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, index):
        img = cv2.imread(self.imgPathList[index], 0)

        # random aug img
        bbox, label = parse_xml((self.imgPathList[index], self.labelPathList[index]))
        img_ori, _, label = randomAug_box(img=img, box=bbox, label=label, is_train=self.is_train)
        resized_im = cv2.resize(img_ori, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(
            'float') * 0.0039216

        # cv2.imshow('img', img)
        # cv2.imshow('img_ori', img_ori)
        # cv2.imshow('resized_im', resized_im)
        # cv2.imshow('img', img)
        # if cv2.waitKey(2000) & 0xFF == ord('q'):
        #     exit

        resized_im = resized_im[np.newaxis,]
        return resized_im, torch.FloatTensor(label), self.imgPathList[index]


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
