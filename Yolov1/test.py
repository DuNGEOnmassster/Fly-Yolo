import torch
import torch.nn as nn
import os

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, 10, requires_grad=True)
target = torch.empty(3, 10, dtype=torch.long).random_(5)
print(input.shape, input)
print(target.shape)
output = loss(input, target)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()

path = r"E:\datasets\VOCdevkit2012\VOC2012\JPEGImages\2007_000027.jpg"

print(os.path.split(path)[-1].split('.')[0])