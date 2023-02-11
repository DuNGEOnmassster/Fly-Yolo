''' train the model '''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time
import math
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data.general import get_anchors
from torch.utils.data import DataLoader

from models.yolov1 import YOLOv1
from data.datasets import YOLODataset, collate_fn
from utils.create_label import CreateTargets
from utils.loss import Criterion

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv1')
    parser.add_argument("--weights_path", default="",
                        help="initial weights path")
    parser.add_argument("--augment", default=False,
                        help="open data augment during training")
    parser.add_argument("--hyp", default="./configure/hyp.yaml",
                        help="the hyperparameter configure of dataset and loss")
    parser.add_argument("--class_names", default="./configure/VOC_classes.yaml",
                        help="the classes of dataset")
    parser.add_argument("--root", default="/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012",
                        help="dataset root path")
    parser.add_argument("--train_path", default="/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt",
                        help="the train dataset path")
    parser.add_argument("--val_path", default="/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012/ImageSets/Main/val.txt",
                        help="the val dataset path")
    parser.add_argument("--test_path", default="/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012/ImageSets/Main/test.txt",
                        help="the test dataset path")
    parser.add_argument("--input_shape", type=int, default="640",
                        help="image size during training")
    parser.add_argument("--epoch", default=50, type=int,
                        help="how many epochs in total for trainning")
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', default=2, type=int,
                        help='The upper bound of warm-up')
    parser.add_argument("--epochs", type=int, default="1",
                        help="training epochs")
    parser.add_argument("--batch_size", type=int, default="4",
                        help="batch size")
    parser.add_argument("--anchors", default="",
                        help="anchors")
    parser.add_argument("--optim", default="sgd",
                        help="optimizer")
    parser.add_argument("--min_lr", type=float, default="0.0001",
                        help="minimum learning rate")
    parser.add_argument("--momentum", type=float, default="0.937",
                        help="momentum of optimizer")
    parser.add_argument("--weight_decay", type=float, default="5e-4",
                        help="weight_decay of optimizer")
    parser.add_argument("--center_sample", action="store_true", default=False,
                        help="open data augment during training")
 
    return parser.parse_args()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # args init
    args = parse_args()
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 存在训练结果的目录
    save_dir = "./logs"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 加载dataset和loss的配置参数
    f_hyp = open(args.hyp, 'r')
    hyp = yaml.load(f_hyp, Loader=yaml.SafeLoader)

    # 加载anchors
    if args.anchors != "":
        f_anchors = open(args.anchors, 'r')
        anchors_cfg = yaml.load(f_anchors, Loader=yaml.SafeLoader)
        anchors = anchors_cfg["anchors"]
        print(anchors)
    else:
        args.anchors = None

    # 加载数据类型名称
    f_classes = open(args.class_names)
    class_names = yaml.load(f_classes, Loader=yaml.SafeLoader)
    class_names = class_names["class_names"]
    num_classes = len(class_names)

    num_detect_layers = 3
    hyp['box'] *= 3. / num_detect_layers  # scale to layers
    hyp['cls'] *= num_classes / 80. * 3. / num_detect_layers  # scale to classes and layers
    hyp['obj'] *= (args.input_shape / 640) ** 2 * 3. / num_detect_layers  # scale to image size and layers


    # model init
    model = YOLOv1(cfg=args, device=device, img_size=args.input_shape, num_classes=20, trainable=True, center_sample=args.center_sample).to(device)

    if args.weights_path == "":
        model.init_bias()
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.weights_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        # 字典的update()
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #   显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示, head部分没有载入是正常现象, Backbone部分没有载入是错误的。\033[0m")

    # 优化器
    # sgd: 学习率lr, 最小学习率min_lr,动量momentum, 正则化权值weight_decay
    epochs = args.epochs
    batch_size = args.batch_size
    base_lr = args.lr
    optimizer = None
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 数据集--训练时数据增强augment
    train_dataset = YOLODataset(args.root, args.train_path, args.augment, class_names, hyp, args.input_shape,
                                batch_size)
    val_dataset = YOLODataset(args.root, args.val_path, not args.augment, class_names, hyp, args.input_shape,
                              batch_size)

    num_workers = 4
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size,
                                num_workers=num_workers)

    # train
    epoch_size = args.epoch
    train_size = val_size = args.input_shape

    for epoch in range(args.epoch):
        with tqdm(range(len(train_dataloader))) as pbar:
            start_time = time.time()
            # use cos lr
            if args.cos and epoch > 20 and epoch <= args.epoch - 20:
                # use cos lr
                tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1. / (args.epoch-20)))
                set_lr(optimizer, tmp_lr)

            elif args.cos and epoch > args.epoch - 20:
                tmp_lr = 0.00001
                set_lr(optimizer, tmp_lr)

            # use step lr
            else:
                tmp_lr = base_lr

            # for iter_i, (images, labels) in enumerate(train_dataloader):
            for iter_i, (images, labels) in zip(pbar, train_dataloader):

                # import pdb; pdb.set_trace()

                bs = images.shape[0]
                # WarmUp strategy for learning rate
                if not args.no_warm_up:
                    if epoch < args.wp_epoch:
                        tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                        # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                        set_lr(optimizer, tmp_lr)

                    elif epoch == args.wp_epoch and iter_i == 0:
                        tmp_lr = base_lr
                        set_lr(optimizer, tmp_lr)

                # to device
                images = images.to(device)

                # multi-scale trick
                if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                    # randomly choose a new size
                    train_size = random.randint(10, 19) * 32
                    model.set_grid(train_size)
                if args.multi_scale:
                    # interpolate
                    images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

                # make labels
                cl = CreateTargets(args.input_shape, args.anchors, model.stride, labels, num_classes)
                targets = cl.create_targets(
                    batch_size=bs,
                    center_sample=args.center_sample)

                obj_pred, cls_pred, bbox_pred = model(images)

                # 损失函数
                criterion = Criterion(cls_pred, obj_pred, bbox_pred, targets, hyp)
                total_loss, loss, loss_cls, loss_obj, loss_bbox = criterion.criterion(train_size, bs)
                # print(f"Epoch{epoch}: iter_i: {iter_i} total_loss:{total_loss}")

                # backprop
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # save model
                if (epoch + 1) % 10 == 0:
                    print('Saving state, epoch:', epoch + 1)
                    torch.save(model.state_dict(), os.path.join(save_dir,
                            'yolo_' + repr(epoch + 1) + '.pth')
                            )

                pbar.set_postfix(
                    Total_Loss=np.round(total_loss.cpu().detach().numpy().item(), 5),
                    CLS_Loss=np.round(loss_cls.cpu().detach().numpy().item(), 5),
                    OBJ_Loss=np.round(loss_obj.cpu().detach().numpy().item(), 5),
                    BBOX_Loss=np.round(loss_bbox.cpu().detach().numpy().item(), 5),
                    EPOCH=epoch
                )
                pbar.update(0)


if __name__ == "__main__":
    train()