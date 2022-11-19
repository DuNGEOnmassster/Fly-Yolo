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


def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # Set offset as 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  
    steps_w = 1.0 / in_width  

    # generate every central point
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # generate “boxes_per_pixel” gourps of h*w 
    # for anchor coordinate(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # get half-height and half-weight
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def get_multibox(image):
    h, w = image.shape[-2:]
    print(h, w)
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)

def detect():
    pass

if __name__ == "__main__":
    args = parse_args()
    image = load_image(args)
    # show_image(img)
    get_multibox(image)