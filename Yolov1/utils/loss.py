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
        self.BCECls = nn.BCEWithLogitsLoss(reduction='none',
                                           pos_weight=torch.tensor([hyp['cls_pw']], device=pred_cls.device))
        self.BCEObj = nn.BCEWithLogitsLoss(reduction='none',
                                           pos_weight=torch.tensor([hyp['obj_pw']], device=pred_cls.device))

    def smooth_label(self, target, eps=0.1):
        return target * (1.0 - 0.5 * eps)

    def criterion(self, img_size, batch_size):
        # 训练时传入预测值的batch_size

        # target_pos(confidence)
        target_pos = self.targets[..., 0]
        # target_bbox
        target_bbox = self.targets[..., 1:5]
        # target_scale
        target_scale = self.targets[..., 5]
        # target_cls
        target_cls = self.targets[..., 6:]

        # 计算边界框IoU时需要将pred_bbox和targets的真实框恢复到原图像大小
        pred_bbox = self.pred_bbox * img_size
        target_bbox = target_bbox * img_size

        # loss_bbox
        ciou = bbox_iou(pred_bbox.mT, target_bbox, x1y1x2y2=False, CIoU=True)
        ciou = ciou.reshape(batch_size, -1)
        # print(ciou.shape)
        loss_bbox = 1.0 - ciou
        loss_bbox = loss_bbox * target_scale
        loss_bbox = (loss_bbox * target_pos).mean()

        # loss_cls
        loss_cls = self.BCECls(self.pred_cls, self.smooth_label(target_cls))
        loss_cls = torch.mean(loss_cls, dim=2)
        # print(loss_cls.shape)
        loss_cls = (loss_cls * target_pos).mean()

        # loss_obj
        target_obj = (1.0 - self.hyp['gr']) + self.hyp['gr'] * ciou.detach().clamp(0).type(target_pos.dtype)
        # print(target_obj.shape)
        loss_obj = self.BCEObj(self.pred_obj, target_obj).mean()

        loss_cls = loss_cls * self.hyp['cls']
        loss_obj = loss_obj * self.hyp['obj']
        loss_bbox = loss_bbox * self.hyp['box']

        loss = loss_cls + loss_obj + loss_bbox

        total_loss = loss * batch_size

        return total_loss, loss, loss_cls, loss_obj, loss_bbox