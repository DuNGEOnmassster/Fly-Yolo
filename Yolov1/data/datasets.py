''' implement the datasets, including make labels and images with data augment '''

'''
    参考
    https://github.com/ultralytics/yolov5
'''

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
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.general import load_img_paths, load_anno_paths, load_img, load_labels, xywhn2xyxy, xyxy2xywh
from utils.path import pathset

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

        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < self.hyp["mixup"]:
                img2, labels2 = self.load_mosaic(random.randint(0, self.length - 1))
                img, labels = self.mixup(img, labels, img2, labels2)

        else:
            # load img
            img, (h0, w0), (h, w) = load_img(img_path=self.img_paths[index], img_size=self.input_shape, augment=self.augment)

            # letterbox
            shape = self.input_shape
            img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = load_labels(class_names=self.class_names, anno_path=self.anno_paths[index], remove_difficult=False).copy()

            # 随机仿射是对坐标的变换, 需要将xywhn2xmin+ymin+xmax+ymax
            # 加灰边填充不存在xy坐标超出图像边界的问题,因此不需要限制xy坐标范围
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # 进行了mosaic(内部有随机仿射),就不需要再进行一次随机仿射
            if not mosaic:
                img, labels = self.random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # hsv的增强
            self.augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        # 标签恢复成yolo格式
        num_labels = len(labels)
        # labels[:, [1, 3]] = np.clip(labels[:, [1, 3]], 0, img.shape[1])
        # labels[:, [2, 4]] = np.clip(labels[:, [2, 4]], 0, img.shape[0])

        # [num_targets, cls_ind + x1 + y1 + x2 + y2] -> [num_targets, cls_ind + nx + ny + nw + nh]
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

        # # 增加图像在一个批量中的批次索引,与一个批量的预测结果保持一致,便于计算loss
        labels_out = torch.zeros((num_labels, 6))
        if num_labels:
            # [num_targets, cls_ind+nx+ny+nw+nh] -> [num_targets, batch_ind+cls_ind+nx+ny+nw+nh]
            labels_out[:, 1:] = torch.from_numpy(labels)

        # (num_targets, cls_idm+x+y+w+h)
        # labels_out = labels

        # img Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        # 任何tensor都要转换成float32类型
        return torch.from_numpy(img).float(), labels_out.float()

    #-----------------------------------------------------------#
    # 数据增强
    #-----------------------------------------------------------#

    # mosaic数据增强方式
    def load_mosaic(self, index):

        yc, xc = (int(random.uniform(-x, 2 * self.input_shape + x)) for x in self.mosaic_border)

        index4 = [index] + random.sample(self.indices, 3)

        labels4 = []

        for i, index in enumerate(index4):
            img, (_, _), (h, w) = load_img(img_path=self.img_paths[index], img_size=self.input_shape, augment=self.augment)
            labels = load_labels(class_names=self.class_names, anno_path=self.anno_paths[index], remove_difficult=False).copy()

            if i == 0:
                img4 = np.full((self.input_shape * 2, self.input_shape * 2, 3), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.input_shape * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, self.input_shape * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(w + xc, self.input_shape * 2), min(yc + h, self.input_shape * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(x2a - x1a, w), min(y2a - y1a, h)
            # h,w
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # 随机仿射是对坐标的变换, 需要将xywhn2xmin+ymin+xmax+ymax
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)

            # 添加格式是xyxy的标签
            labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)

        # 限制xy坐标范围, mosaic可能出现xy坐标超出图像边界的问题
        for x in labels4[:, 1:]:
            np.clip(x, 0, self.input_shape * 2, out=x)

        # Augment
        img4, labels4 = self.random_perspective(img4, labels4,
                                                degrees=self.hyp['degrees'],
                                                translate=self.hyp['translate'],
                                                scale=self.hyp['scale'],
                                                shear=self.hyp['shear'],
                                                perspective=self.hyp['perspective'],
                                                border=self.mosaic_border)

        return img4, labels4

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

    def mixup(self, img1, label1, img2, label2):
        r = np.random.beta(32.0, 32.0)
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((label1, label2), 0)
        return img, labels

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

# 在这里将label改为[num_targets, batch_ind+cls_ind_+nx+ny+nw+nh]
def collate_fn(batch):
    __doc__ = r"""
        parms:
           batch tuple (imgs, labels)
       return:
           imgs tensor
           labels list of tensor [num_targets, batch_ind+cls_ind+nx+ny+nw+nh]
       """
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)


if __name__ == "__main__":
    chosen_pathset = "pathset2"
    root = pathset[chosen_pathset]["root"]
    train_path = pathset[chosen_pathset]["train_path"]
    f_hyp = open("configure/hyp.yaml", 'r')
    # yaml将文件load成字典
    hyp = yaml.load(f_hyp, Loader=yaml.SafeLoader)
    class_names = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]

    train_dataset = YOLODataset(root, train_path, augment=True, class_names=class_names, hyp=hyp, input_shape=640, batch_size=2)
    batch_size = 8
    num_workers = 4
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers)

    for i, (img, labels) in enumerate(train_dataloader):
        print(img.shape)
        print(labels)

    # img, labels, shapes = train_dataset[0]
    #
    # img = img.numpy()
    # labels = labels.numpy()
    # img = img.transpose(1, 2, 0)
    #
    # h, w = img.shape[:2]
    # labels[:, 2:] = xywhn2xyxy(labels[:, 2:], w, h)
    #
    # pts = labels[:, 2:]
    #
    # img1 = Image.fromarray(img).convert("RGB")
    # draw = ImageDraw.Draw(img1)
    #
    # for pt in pts:
    #     draw.polygon([(pt[0], pt[1]), (pt[2], pt[1]), (pt[2], pt[3]), (pt[0], pt[3])], outline=(255,0,0))
    # del draw
    #
    # img1 = np.array(img1)
    # # plt.imshow(img1)
    # # plt.show()
    #
    # img1 = img1[:, :, ::-1]
    # cv2.imwrite(r'E:\datasets\yolo_dataset\test1.png', img1)




