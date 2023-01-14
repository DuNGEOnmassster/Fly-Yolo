import torch
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 图片导入函数
def load_image(img_path, img_height=None, img_width=None):
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


# content归一化
def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image
