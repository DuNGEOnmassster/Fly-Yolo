"""制作样本"""
import numpy as np
import torch
import torch.nn as nn
from data.general import bbox_iou

__doc__ = r"""
类CreateTargets

输入变量
image_size(tuple) = (640, 640)
anchors(list) = [   [10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]  ]
strides(list) = [8, 16, 32]

labels(tensor) = [num_targets, batch_ind+cls_ind+nx+ny+nw+nh]

函数
set_anchors: 将anchors的类型转成tensor
calculate_IoU: 计算一个真实框和全部锚框之间的IoU
build_target_without_anchors: 不基于锚框，制作一个正样本的position_target
build_target_with_anchors: 基于锚框，制作一个正样本的position_target
create_targets: 制作全部样本

"""

class CreateTargets():
    def __init__(self, image_size, anchors, strides, labels):
        super(CreateTargets, self).__init__()
        """
        初始化
        """
        self.image_size = image_size
        self.anchors = anchors
        self.strides = strides
        self.labels = labels
    
    def set_anchors(self):
        anchors = np.array(self.anchors)
        temp = np.zeros((anchors.shape[0], 4))
        temp[:, 2:] = anchors
        temp_anchors = torch.from_numpy(temp).float()
        return temp_anchors
    
    def calculate_iou(self, target, temp_anchors, x1y1x2y2=True):
        iou = bbox_iou(target.T, temp_anchors, x1y1x2y2=x1y1x2y2)
        return iou

    def build_target_without_anchors(self, target):
        stride_ind = 0
        anchor_ind = 0
        img_h, img_w = self.image_size

        batch_ind, _, nx, ny, nw, nh = target

        xc_s = nx * img_w / self.strides[stride_ind]
        yc_s = ny * img_h / self.strides[stride_ind]
        grid_x = int(xc_s)
        grid_y = int(yc_s)

        return [int(stride_ind), int(batch_ind), int(anchor_ind), grid_x, grid_y]

    def build_target_with_anchors(self):
        pass

    def create_targets(self, center_sample=False):

        img_h, img_w = self.image_size
        num_of_batch_ind = self.labels.shape[0]
        y_trues = []
        num_stride = len(self.strides)
        num_anchor_of_one_layer = len(self.anchors) // num_stride if self.anchors is not None else 1

        for s in self.strides:
            fmp_w = img_w // s
            fmp_h = img_h // s
            y_trues.append(np.zeros([num_of_batch_ind, num_anchor_of_one_layer, fmp_h, fmp_w, 1+1+4+1]))

        for i, label in enumerate(self.labels):
            label = label.numpy()
            cls_ind = label[1]
            # 平衡因子box_scale: 小框给大权重，大框给小权重
            box_scale = 2.0 - (label[4] / img_w) * (label[5] / img_h)
            position_target = []
            if self.anchors is not None:
                pass
            else:
                position_target = self.build_target_without_anchors(label)

            stride_ind, batch_ind, anchor_ind, grid_x, grid_y = position_target

            if center_sample:
                # We consider four grid points near the center point
                for j in range(grid_y, grid_y + 2):
                    for i in range(grid_x, grid_x + 2):
                        if (j >= 0 and j < y_trues[stride_ind].shape[2]) and (
                                i >= 0 and i < y_trues[stride_ind].shape[3]):
                            # confidence
                            y_trues[stride_ind][batch_ind, anchor_ind, j, i, 0] = 1.0
                            # class_ind
                            y_trues[stride_ind][batch_ind, anchor_ind, j, i, 1] = cls_ind
                            # nx+ny+nw+nh
                            y_trues[stride_ind][batch_ind, anchor_ind, j, i, 2:6] = label[2:]
                            # box_scale
                            y_trues[stride_ind][batch_ind, anchor_ind, j, i, 6] = box_scale
            else:
                # We only consider top-left grid point near the center point
                if (grid_y >= 0 and grid_y < y_trues[stride_ind].shape[2]) and (
                        grid_x >= 0 and grid_x < y_trues[stride_ind].shape[3]):
                    # confidence
                    y_trues[stride_ind][batch_ind, anchor_ind, grid_y, grid_x, 0] = 1.0
                    # class_ind
                    y_trues[stride_ind][batch_ind, anchor_ind, grid_y, grid_x, 1] = cls_ind
                    # nx+ny+nw+nh
                    y_trues[stride_ind][batch_ind, anchor_ind, grid_y, grid_x, 2:6] = label[2:]
                    # box_scale
                    y_trues[stride_ind][batch_ind, anchor_ind, grid_y, grid_x, 6] = box_scale

            y_trues = [y_true.reshape(num_of_batch_ind, -1, 1 + 1 + 4 + 1) for y_true in y_trues]
            y_trues = np.concatenate(y_trues, axis=1)

            return torch.from_numpy(y_trues).float()

if __name__ == "__main__":
    anchors = [   [10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]  ]
    anchor_strides = [8, 16, 32]
    no_anchor_strides = [32]
    target_wh = torch.tensor([0, 0, 30, 30])
    target = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])

    create_targets = CreateTargets(image_size=(320, 320), anchors=None, strides=no_anchor_strides, labels=target)

    # temp_anchors = create_targets.set_anchors()
    # iou = create_targets.calculate_iou(target_wh, temp_anchors, x1y1x2y2=False)
    y_trues = create_targets.create_targets(center_sample=True)
    print(y_trues, y_trues.shape, y_trues[y_trues[..., 0] == 1.0])

    # print(temp_anchors, temp_anchors.shape, temp_anchors.dtype)
    # print(iou)
    print(target, target.shape)

