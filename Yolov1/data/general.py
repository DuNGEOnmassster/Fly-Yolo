import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path import pathset

def load_img_paths(root, path):
    with open(path, 'r') as f:
        items = f.readlines()

    img_names = []
    for item in items:
        img_name = item.split()
        img_names.append(img_name[0])

    postfix_txt = os.path.join(os.path.split(path)[0], 'postfix.txt')
    with open(postfix_txt, 'r') as f:
        postfix = f.read()

    img_root = os.path.join(root, 'JPEGImages')
    img_paths = [os.path.join(img_root, str(i)+postfix) for i in img_names]
    return img_paths

def load_anno_paths(root, path):
    with open(path, 'r') as f:
        items = f.readlines()

    img_names = []
    for item in items:
        img_name = item.split()
        img_names.append(img_name[0])


    anno_root = os.path.join(root, 'Annotations')
    anno_paths = [os.path.join(anno_root, str(i)+'.xml') for i in img_names]
    return anno_paths

# 获得图像
def load_img(img_path, img_size, augment):

    assert img_path is not None, "图像不存在"

    img = cv2.imread(img_path)
    h0, w0 = img.shape[0], img.shape[1]
    r = img_size / max(h0, w0)
    if r !=1:
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0*r), int(h0*r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]


# 获得类别下标和yolo格式标签(num_labels, cls_ind+nx+ny+nw+nh)
def load_labels(class_names, anno_path, remove_difficult=False):

    target = ET.parse(anno_path)
    root = target.getroot()

    # 获得高和宽
    size = root.find("size")
    h = int(size.find("height").text)
    w = int(size.find("width").text)

    labels = []
    for object in root.iter("object"):
        difficult = int(object.find("difficult").text) == 1
        if difficult and remove_difficult:
            continue
        cls_name = object.find("name").text.strip()
        cls_index = int(class_names.index(cls_name))

        bndbox = object.find("bndbox")
        bbox = []
        points = ['xmin', 'ymin', 'xmax', 'ymax']
        for point in points:
            pt = float(bndbox.find(point).text)
            bbox.append(pt)
        label = [cls_index] + bbox
        labels.append(label)

    labels = np.array(labels, dtype=np.float32)
    y = np.copy(labels)
    y[:, 1] = ((labels[:, 1] + labels[:, 3]) / 2) / w
    y[:, 2] = ((labels[:, 2] + labels[:, 4]) / 2) / h
    y[:, 3] = np.abs((labels[:, 1] - labels[:, 3])) / w
    y[:, 4] = np.abs((labels[:, 2] - labels[:, 4])) / h

    return y

def get_anchors(anchors):
    anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = anchors.reshape(-1, 2)
    return anchors

# # 获得yolo格式的标签
# def xyxy2xywhn(bboxes, w, h):
#     y = np.copy(bboxes)
#     y[:, 0] = ((bboxes[:, 0] + bboxes[:, 2]) / 2) / w
#     y[:, 1] = ((bboxes[:, 1] + bboxes[:, 3]) / 2) / h
#     y[:, 2] = np.abs((bboxes[:, 0] - bboxes[:, 2])) / w
#     y[:, 3] = np.abs((bboxes[:, 1] - bboxes[:, 3])) / h
#     return y

# xywhn2xmin+ymin+xmax+ymax
def xywhn2xyxy(x, w, h, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
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


if __name__ == "__main__":
    chosen_pathset = "pathset2"

    root = pathset[chosen_pathset]["root"]
    train_path = pathset[chosen_pathset]["train_path"]

    class_names = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]

    img_paths = load_img_paths(root, train_path)
    anno_paths = load_anno_paths(root, train_path)
    labels = load_labels(class_names, anno_paths[2])

    for label in labels:
        print(label)

    img = cv2.imread(img_paths[2])
    h, w = img.shape[:2]
    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h)
    # print(img_paths)
    # print(anno_paths)
    for label in labels:
        print(label)


