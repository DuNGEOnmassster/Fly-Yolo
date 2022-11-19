import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Anchor attemption")

    parser.add_argument("--image_path", type=str, default="./dataset/catdog.jpg",
                        help="declare image restore path")
    parser.add_argument("--max_size", type=int, default=400,
                        help="declare the maximum size of image")
    parser.add_argument("--set_size", type=int, default=False,
                        help="declare specific image size if required")
    
    return parser.parse_args()


def load_image(args):
    image = Image.open(args.image_path).convert("RGB")

    if args.set_size:
        size = int(input("Input require image size, which shouldn't over 400"))
    
    if max(image.size) > args.max_size:
        size = args.max_size
    else:
        size = max(image.size)

    # Transform image into tensor and normalize tensor
    trans_image = transforms.Compose(
        [transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))]
    )

    image = trans_image(image)[:3, :, :].unsqueeze(dim=0)
    return image

def detect():
    pass

if __name__ == "__main__":
    args = parse_args()
    img = load_image(args)
    print(img)