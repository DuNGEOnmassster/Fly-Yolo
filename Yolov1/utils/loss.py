"""计算损失"""
import torch
import torch.nn as nn
from data.general import bbox_iou

class Criterion():
    def __init__(self, pred_cls, pred_obj, pred_bbox, targets, hyp):
        super(Criterion, self).__init__()
        self.pred_cls = pred_cls
        self.pred_obj = pred_obj
        self.pred_bbox = pred_bbox
        self.targets = targets.float().to(pred_cls.device)
        self.hyp = hyp
        # self.BCECls = nn.BCEWithLogitsLoss(reduction='none',
        #                                    pos_weight=torch.tensor([hyp['cls_pw']], device=pred_cls.device))
        self.BCECls = nn.CrossEntropyLoss(reduction='none')
        self.BCEObj = nn.BCEWithLogitsLoss(reduction='none')

    def smooth_label(self, target, eps=0.1):
        return target * (1.0 - 0.5 * eps)

    def scale_loss(self, loss, batch_size, num_pos):
        if self.hyp['scale_loss'] == 'batch':
            # scale loss by batch size
            loss = loss.sum() / batch_size
        elif self.hyp['scale_loss'] == 'positive':
            # scale loss by number of positive samples
            loss = loss.sum() / num_pos

        return loss

    def criterion(self, img_size, batch_size):
        # 训练时传入预测值的batch_size

        # target_pos(confidence)
        target_pos = self.targets[..., 0].float()
        # target_bbox nxnynwnh
        target_bbox = self.targets[..., 1:5].float()
        # target_scale
        target_scale = self.targets[..., 5].float()
        # target_cls
        target_cls = self.targets[..., 6].long()

        # 正样例的数量
        num_pos = target_pos.sum().clamp(1.0)

        # 计算边界框IoU时需要将pred_bbox和targets的真实框恢复到原图像大小
        pred_bbox = self.pred_bbox * img_size
        target_bbox = target_bbox * img_size

        # target = self.targets[self.targets[..., 0] == 1]
        # print(f"target_bbox:{target * img_size}")

        # loss_bbox
        ciou = bbox_iou(pred_bbox.T, target_bbox, x1y1x2y2=False, CIoU=True)
        ciou = ciou.reshape(batch_size, -1)
        # print(ciou.shape)
        loss_bbox = 1.0 - ciou
        loss_bbox = loss_bbox * target_scale
        loss_bbox = loss_bbox * target_pos
        loss_bbox = self.scale_loss(loss_bbox, batch_size, num_pos)

        # loss_cls
        # [B, HW, C] -> [B, C, HW]
        pred_cls = self.pred_cls.permute(0, 2, 1)
        loss_cls = self.BCECls(pred_cls, target_cls)
        # print(loss_cls.shape)
        loss_cls = loss_cls * target_pos
        loss_cls = self.scale_loss(loss_cls, batch_size, num_pos)

        # loss_obj
        target_obj = (1.0 - self.hyp['gr']) + self.hyp['gr'] * ciou.detach().clamp(0).type(target_pos.dtype)
        # print(target_obj.shape)
        loss_obj = self.BCEObj(self.pred_obj, target_obj)
        loss_obj = self.scale_loss(loss_obj, batch_size, num_pos)

        loss_cls = loss_cls * self.hyp['cls']
        loss_obj = loss_obj * self.hyp['obj']
        loss_bbox = loss_bbox * self.hyp['box']

        total_loss = loss_cls + loss_obj + loss_bbox

        return total_loss, loss_cls, loss_obj, loss_bbox
