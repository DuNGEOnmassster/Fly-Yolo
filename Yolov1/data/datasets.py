''' implement the datasets, including make labels and images with data augment '''

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tqdm
import random
import os
import math
import yaml
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

from data.general import load_img_paths, load_anno_paths, load_img, load_labels, xywhn2xyxy, xyxy2xywh


class YOLODataset(Dataset):

    def __init__(self, root, path, augment, class_names=None, hyp=None, input_shape=640, batch_size=16, stride=32, pad=0.0):
        super(YOLODataset, self).__init__()

        self.root = root
        self.path = path
        self.augment = augment
        self.batch_size = batch_size
        self.mosaic = self.augment
        self.input_shape = input_shape
        self.class_names = class_names
        self.hyp = hyp
        self.img_paths = load_img_paths(self.root, self.path)
        self.anno_paths = load_anno_paths(self.root, self.path)
        self.nc = len(class_names)
        self.length = len(self.img_paths)
        self.stride = stride
        self.pad = pad

        self.mosaic_border = [-input_shape // 2, -input_shape // 2]
        self.indices = range(self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp

        if self.augment:
            # load img
            img, (h0, w0), (h, w) = load_img(img_path=self.img_paths[index], img_size=self.input_shape,
                                             augment=self.augment)
            labels = load_labels(class_names=self.class_names, anno_path=self.anno_paths[index],
                                 remove_difficult=False).copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h)

            img, labels = self.random_perspective(img, labels,
                                                  degrees=hyp['degrees'],
                                                  translate=hyp['translate'],
                                                  scale=hyp['scale'],
                                                  shear=hyp['shear'],
                                                  perspective=hyp['perspective'])

            # hsv的增强
            self.augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        else:
            # load img
            img, (h0, w0), (h, w) = load_img(img_path=self.img_paths[index], img_size=self.input_shape, augment=self.augment)

            # letterbox
            shape = self.input_shape
            img, ratio, pad = self.letterbox(img, shape, auto=True, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = load_labels(class_names=self.class_names, anno_path=self.anno_paths[index], remove_difficult=False).copy()

            # 随机仿射是对坐标的变换, 需要将xywhn2xmin+ymin+xmax+ymax
            # 加灰边填充不存在xy坐标超出图像边界的问题,因此不需要限制xy坐标范围
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])


        # 标签恢复成yolo格式
        num_labels = len(labels)
        if num_labels:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if num_labels:
                    # 归一化y翻转
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if num_labels:
                    # 归一化x翻转
                    labels[:, 1] = 1 - labels[:, 1]

        # (num_targets, cls_idm+x+y+w+h)
        labels_out = labels

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        return img, labels_out

    #-----------------------------------------------------------#
    # 数据增强方式: random_perspective, augment_hsv
    #-----------------------------------------------------------#

    # 通过设定高、宽的阈值，设定高宽比的阈值，设定区域面积比的阈值来筛选可以使用的框
    # eps是为了防止除以0
    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

    # 随机仿射数据增强方式
    # 一般perspective: 0.0均设为0.0
    def random_perspective(self, img, targets=(), segments=(), degrees=10., translate=.1, scale=.1, shear=10., perspective=0.0,
                           border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:  # warp segments
                pass

            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n,
                                                                                    8)  # perspective rescale or affine

                # create new horizontal boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle，保持原高宽比缩放图像
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch，直接缩放图像
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# 在这里将label改为[num_targets, img_ind+cls_ind_+x+y+w+h]
def collate_fn(batch):
    imgs = []
    labels = []

    for img, label in batch:
        imgs.append(img)
        labels.append(label)

    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    labels = [torch.from_numpy(np.array(label)).type(torch.FloatTensor) for label in labels]

    return imgs, labels

if __name__ == "__main__":
    root = r"E:\datasets\yolo_dataset"
    train_path = r"E:\datasets\yolo_dataset\ImageSets\Main\train.txt"
    f_hyp = open("../configure/hyp.yaml", 'r')
    # yaml将文件load成字典
    hyp = yaml.load(f_hyp, Loader=yaml.SafeLoader)
    class_names = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]

    train_dataset = YOLODataset(root, train_path, augment=False, class_names=class_names, hyp=hyp)

    img, labels = train_dataset[2]

    img = img
    labels = labels
    img = img.transpose(1, 2, 0)

    h, w = img.shape[:2]
    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h)

    pts = labels[:, 1:]

    img1 = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(img1)

    for pt in pts:
        draw.polygon([(pt[0], pt[1]), (pt[2], pt[1]), (pt[2], pt[3]), (pt[0], pt[3])], outline=(255,0,0))
    del draw

    img1 = np.array(img1)
    # plt.imshow(img1)
    # plt.show()

    img1 = img1[:, :, ::-1]
    cv2.imwrite(r'E:\datasets\yolo_dataset\test1.png', img1)
