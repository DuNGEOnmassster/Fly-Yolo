import torch
import torch.nn as nn
from torchsummary import summary

# darknet basic component
class Block(nn.Module):
    def __init__(self, channel=64):
        super(Block, self).__init__()
        in_channel = channel // 2
        self.Conv1 = nn.Conv2d(in_channels=channel, out_channels=in_channel, kernel_size=1)
        self.Conv2 = nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.Conv2(self.Conv1(x))
        return x + y

class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_*(len(k) + 1), c2, 1, 1)
        self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=i, padding=i//2, stride=1) for i in k])
    
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x]+[m(x) for m in self.maxpool], 1))

if __name__ == "__main__":
    spp = SPP(1024, 512)
    summary(model=spp, input_size=(1024, 8, 8), device="cpu", batch_size=1)
