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


def get_convert(tensor):
    # delete batch dim
    image = tensor.data.numpy().squeeze()
    # transpose [c,h,w] -> [h,w,c]
    image = image.transpose(1,2,0)
    # imverse normalization
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def show_image(image):
    print(f"image shape: {image.shape}")
    fig, (ax1) = plt.subplots(1,figsize=(16, 16))
    ax1.imshow(get_convert(image))
    ax1.set_title('origin image')
    ax1.axis('off')
    plt.show()


def detect():
    pass

if __name__ == "__main__":
    args = parse_args()
    img = load_image(args)
    show_image(img)