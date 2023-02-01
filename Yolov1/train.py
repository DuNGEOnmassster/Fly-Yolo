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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
# from pytorchtools import EarlyStopping
import tools

from models.darknet53 import Darknet53
from utils.process_dataset import load_data

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv1')
    parser.add_argument("-d", "--dataset", default="/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012/JPEGImages", 
                        help="folder where origin input img data set")
    parser.add_argument("-o", "--model_weight", default="./../model_weight/",
                        help="folder where to save model after trainning")

    parser.add_argument("--train_split", default=0.8, type=float,
                        help="how much data is used for trainning")
    parser.add_argument("--valid_split", default=0.1, type=float,
                        help="how much data is used for validing")
    parser.add_argument("--epoch", default=50, type=int,
                        help="how many epochs in total for trainning")
    parser.add_argument("--batch_size_train", type=int, default=64,
                        help="declare batch size of train_loader")
    parser.add_argument("--batch_size_test", type=int, default=1000,
                        help="declare batch size of test_loader")
    # parser.add_argument("--valid_split", type=float, default=0.3,
    #                     help="declare proportion of valid in test_loader split")

    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', default=2, type=int, 
                        help='The upper bound of warm-up')   
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument("--random_seed", type=int, default=3407,
                        help="declare random seed")
 
    return parser.parse_args()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # args init
    args = parse_args()
    # model init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet53()
    # load_data
    train_loader, valid_loader, test_loader = load_data(args.batch_size, args.train_split, args.valid_split, device, args)

    print(train_loader.dataset)
    print(valid_loader.dataset.dataset, "\nLength of valid loader: ", len(valid_loader))
    print(test_loader.dataset.dataset, "\nLength of test loader: ", len(test_loader))
    print(valid_loader.dataset[1][1])
    print(test_loader.dataset[1][1])
    # tricks init
    criterion = nn.CrossEntropyLoss()
    # early_stopping = EarlyStopping(args.patient, verbose=False)
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )
    clr = CosineAnnealingLR(optimizer, T_max=15000)
    # 
    if args.multi_scale: # for example
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]
    
    # train
    for epoch in args.epoch:
        start_time = time.time()
        # use cos lr
        if args.cos and epoch > 20 and epoch <= args.epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (args.epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > args.epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            tmp_lr = base_lr

        for iter_i, (images, targets) in enumerate(dataloader):
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
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
        
            # make train label
            # 存疑！！！
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=train_size, stride=model.stride, label_lists=targets)
            targets = torch.tensor(targets).float().to(device)
            
            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # save model
            if (epoch + 1) % 10 == 0:
                print('Saving state, epoch:', epoch + 1)
                torch.save(model.state_dict(), os.path.join(args.m, 
                        'yolo_' + repr(epoch + 1) + '.pth')
                        ) 


if __name__ == "__main__":
    train()