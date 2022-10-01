import torch
import torch.nn as nn

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