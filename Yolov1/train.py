import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv1')
    # 全没了，晚上再写
    parser.add_argument("-d", "--dataset", default="./../img_data/", 
                        help="folder where origin input img data set")
    parser.add_argument("-o", "--model_weight", default="./../model_weight/",
                        help="folder where to save model after trainning")

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

