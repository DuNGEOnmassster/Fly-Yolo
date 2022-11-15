import torch
from torch import nn
from torchsummary import summary
from common import Block

class Darknet53(nn.Module):
    def __init__(self, channel=64, stages=[1, 2, 8, 8, 4]):
        super(Darknet53, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=channel//2, kernel_size=3)
        self.Conv2 = nn.Conv2d(in_channels=channel//2, out_channels=channel, kernel_size=3, stride=2)
        self.Stage1 = nn.Sequential(*[Block(channel) for i in range(stages[0])])
        self.Conv3 = nn.Conv2d(in_channels=channel, out_channels=channel*2, kernel_size=3, stride=2)
        self.Stage2 = nn.Sequential(*[Block(channel*2) for i in range(stages[1])])
        self.Conv4 = nn.Conv2d(in_channels=channel*2, out_channels=channel*4, kernel_size=3, stride=2)
        self.Stage3 = nn.Sequential(*[Block(channel*4) for i in range(stages[2])])
        self.Conv5 = nn.Conv2d(in_channels=channel*4, out_channels=channel*8, kernel_size=3, stride=2)
        self.Stage4 = nn.Sequential(*[Block(channel*8) for i in range(stages[3])])
        self.Conv6 = nn.Conv2d(in_channels=channel*8, out_channels=channel*16, kernel_size=3, stride=2)
        self.Stage5 = nn.Sequential(*[Block(channel*16) for i in range(stages[4])])


    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Stage1(x)
        x = self.Conv3(x)
        x = self.Stage2(x)
        x = self.Conv4(x)
        x = self.Stage3(x)
        x = self.Conv5(x)
        x = self.Stage4(x)
        x = self.Conv6(x)
        y = self.Stage5(x)
        return y
        

if __name__ == "__main__":
    model = Darknet53()
    summary(model, input_size=(3, 320, 320), batch_size=1, device="cpu")
    # x = torch.rand((1,3,448,448), dtype=torch.float32)
    # y = model(x)
    # print(y)




        
        




