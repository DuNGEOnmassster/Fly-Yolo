''' implement Yolov1 '''

import numpy as np
import torch
import torch.nn as nn
import torchvision
from darknet53 import Darknet53
from common import SPP
from torchsummary import summary


class YOLOv1(nn.Module):
    def __init__(self,
                cfg=None,
                device=None,
                num_classes=20,
                trainable=False,
                center_sample=False):
        super(YOLOv1, self).__init__()
        self.cfg = cfg
        self.devive = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.center_sample = center_sample

        # constants
        self.feature_dim = 512
        self.stride = 32

        # backbone
        self.backbone = Darknet53(channel=32)
        # nack head
        self.nack = nn.Sequential(
            SPP(self.feature_dim, self.feature_dim)
        )
        # detect head
        self.cls_feat = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
        )

        self.reg_feat = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
            nn.Conv2d(self.feature_dim, self.feature_dim, k=3, p=1, s=1),
        )

        self.obj_pred = nn.Conv2d(self.feature_dim, 1, k=1)
        self.cls_pred = nn.Conv2d(self.feature_dim, self.num_classes, k=1)
        self.reg_pred = nn.Conv2d(self.feature_dim, 4, k=1)

        if self.trainable:
            self.init_bias()


    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)
    
    def set_grids(self, img_size):
        self.img_size = img_size
        self.grid_xy = self.create_grids(img_size=img_size)
    
    def create_grids(self, img_size):
        __doc__ = r"""
        parm:
            img_size tuple (img_size, img_size)
            grid_xy tensor [1, HW, 2(x+y)]
        """
        fmp_h, fmp_w = img_size // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        grid_xy = torch.cat([grid_x, grid_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [1, HW, 2]
        grid_xy = grid_xy.unsqueeze(0).to(self.devive)
        return grid_xy
    
    def decode_box(self, reg_pred):
        __doc__ =r"""
        parm: 
            reg_pred tensor [B, HW, 4(tx+ty+tw+th)]
        return:
            bbox_pred tensor [B, HW, 4(x1+y1+x2+y2)]
        """
        # [B, HW, 4] (tx+ty+th+tw) -> (x1+y1+x2+y2)
        # [B, HW, tx+ty] -> [B, HW, x+y] 
        if self.center_sample:
            xy_pred = reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy
        else:
            xy_pred = reg_pred[..., :2].sigmoid() + self.grid_xy
        
        # [B, HW, tw+th] -> [B, HW, w+h]
        wh_pred = reg_pred[..., 2:].exp()

        # xywh -> x1y1x2y2
        x1y1_pred = xy_pred - wh_pred / 2
        x2y2_pred = xy_pred + wh_pred / 2
        bbox_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        # 上采样到输入图像大小
        bbox_pred = bbox_pred * self.stride

        return bbox_pred
    
    def reference(self, x):
        __doc__ = r"""
        推理函数，获得单张图像的边界框和类别预测值
        parm:
            x tensor [1, 3, H, W]
        return:
            obj_pred tensor [B, HW, 1(obj_score)]
            cls_pred tensor [B, HW, C(num_classes)]
            bbox_pred tensor [B, HW, 4(normalized_x1+y1+x2+y2)]
        """
        C = self.num_classes

        x = self.backbone(x)

        x = self.nack(x)
        
        cls_feat = self.cls_feat(x)
        reg_feat = self.reg_feat(x)
        # pred
        obj_pred = self.obj_pred(reg_feat)
        reg_pred = self.reg_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)

        # [1, 1, H, W] -> [1, H, W, 1] -> [1, HW, 1]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(1, -1, 1)
        # [1, C, H, W] -> [1, H, W, C] -> [1, HW, C]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(1, -1, C)
        # [1, 4, H, W] -> [1, H, W, 4] -> [1, HW, 4]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        # txtytwth -> x1y1x2y2
        bbox_pred = self.decode_box(reg_pred=reg_pred)
        # x1y1x2y2 -> normalized_x1y1x2y2
        bbox_pred = bbox_pred / self.img_size

        return obj_pred, cls_pred, bbox_pred

        
    def forward(self, x):
        __doc__ = r"""
        训练函数，拟合真实框和边界框
        parm: 
            x tensor [B, 3, H, W]
        return:
            obj_pred tensor [B, HW, 1(obj_score)]
            cls_pred tensor [B, HW, C(num_classes)]
            bbox_pred tensor [B, HW, 4(normalized_x1+y1+x2+y2)]
        """
        if not self.trainable:
            self.reference(x)
        else:
            B = x.shape[0]
            C = self.num_classes

            x = self.backbone(x)

            x = self.nack(x)
            
            cls_feat = self.cls_feat(x)
            reg_feat = self.reg_feat(x)
            # pred
            obj_pred = self.obj_pred(reg_feat)
            reg_pred = self.reg_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)

            # [B, 1, H, W] -> [B, H, W, 1] -> [B, HW, 1]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, C, H, W] -> [B, H, W, C] -> [B, HW, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, 4, H, W] -> [B, H, W, 4] -> [B, HW, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

            # txtytwth -> x1y1x2y2
            bbox_pred = self.decode_box(reg_pred=reg_pred)
            # x1y1x2y2 -> normalized_x1y1x2y2
            bbox_pred = bbox_pred / self.img_size

        return obj_pred, cls_pred, bbox_pred
