''' implement the VOC and COCO evaluate function, including precision, recall, map and so on '''
import cv2
import os
import numpy as np
import torch
from data.general import load_img_paths, load_anno_paths, load_img, load_labels, xywhn2xyxy, xyxy2xywh, letterbox


class Evaluater():

    def __init__(self,
                 input_shape,
                 anchors,
                 strides,
                 class_names,
                 num_classes,
                 root,
                 path,
                 device,
                 num_thresh,
                 conf_thresh
                 ):
        super(Evaluater, self).__init__()
        self.input_shape = input_shape
        self.anchors = anchors
        self.class_names = class_names
        self.num_classes = num_classes
        self.img_paths = load_img_paths(root, path)
        self.anno_paths = load_anno_paths(root, path)
        self.lengths = len(self.img_paths)
        self.strides = strides
        self.device = device
        self.nms_thresh = num_thresh
        self.conf_thresh = conf_thresh

    '''
    生成一幅图像全部预测框(class_name, x, y, w, h)的txt
    '''
    def get_detections_txt(self, model):

        if os.path.exists("./detections") is False:
            os.makedirs("./detections")
        if os.path.exists("./groundtruths") is False:
            os.makedirs("./groundtruths")
        print("get map.")

        for i in range(self.lengths):
            # load img
            img, (h0, w0), (h, w) = load_img(self.img_paths[i], self.input_shape, False)

            # load groundtruth, nxnynwnh
            gt = load_labels(self.class_names, self.anno_paths[i], False).copy()

            # letterbox
            shape = self.input_shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=False)

            if gt.size:
                gt[:, 1:] = xywhn2xyxy(gt[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()

            # expand batch
            img = torch.unsqueeze(img, 0)

            # print(img.shape)

            with torch.no_grad():
                if self.device:
                    img = img.to(self.device)

                obj_pred, cls_pred, bbox_pred = model(img)
                obj_pred = obj_pred[0]
                cls_pred = cls_pred[0]
                bbox_pred = bbox_pred[0]

                # nxnynwnh -> nx1ny1nx2ny2
                bbox_pred[..., 0] = bbox_pred[..., 0] - bbox_pred[..., 2] / 2
                bbox_pred[..., 1] = bbox_pred[..., 1] - bbox_pred[..., 3] / 2
                bbox_pred[..., 2] = bbox_pred[..., 0] + bbox_pred[..., 2] / 2
                bbox_pred[..., 3] = bbox_pred[..., 1] + bbox_pred[..., 3] / 2

                # scores
                scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)

                # to cpu
                scores = scores.to('cpu').numpy()
                bbox_pred = bbox_pred.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(bbox_pred, scores)

                bboxes = bboxes * self.input_shape

                detections_root = r"E:\workspace\PycharmProjects\Fly-Yolo\Yolov1\detections"
                detection_name = os.path.split(self.img_paths[i])[-1].split('.')[0]

                detection_path = os.path.join(detections_root, detection_name + '.txt')
                # print(detection_path)

                with open(detection_path, "w") as f:
                    for j, box in enumerate(bboxes):
                        # box (ymin, xmin, ymax, xmax)
                        xmin, ymin, xmax, ymax = bboxes[j]
                        x = (xmin + xmax) / 2
                        y = (ymin + ymax) / 2
                        w = (xmax - xmin)
                        h = (ymax - ymin)
                        label_box = [x, y, w, h]

                        str1 = self.class_names[cls_inds[j]] + " " + scores[j] + " ".join(str(k) for k in label_box)
                        # print(str1)
                        f.write(str1 + "\n")
                f.close()

    '''
    生成一幅图像全部真实框(class_name, x, y, w, h)的txt
    '''
    def get_gt_txt(self, ):
        pass

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

if __name__ == "__main__":

    pass
