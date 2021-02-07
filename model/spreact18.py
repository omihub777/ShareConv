import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import ConvBlock, SPreActBlock

class SPreAct18(nn.Module):
    def __init__(self, in_c, num_classes, bias=False):
        super(SPreAct18, self).__init__()
        self.blc1 = ConvBlock(3, 64)

        self.blc2 = nn.Sequential(
            SPreActBlock(64, 64, bias=bias),
            SPreActBlock(64, 64, bias=bias)
        )

        self.blc3 = nn.Sequential(
            SPreActBlock(64, 128, s=2, bias=bias),
            SPreActBlock(128, 128, bias=bias)
        )

        self.blc4 = nn.Sequential(
            SPreActBlock(128, 256, s=2, bias=bias),
            SPreActBlock(256, 256, bias=bias)
        )

        self.blc5 = nn.Sequential(
            SPreActBlock(256, 512, s=2, bias=bias),
            SPreActBlock(512, 512, bias=bias)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    # blc = SPreActBlock(c, 16, k=3, s=1, p=1)
    net = SPreAct18(in_c=c, num_classes=10)
    torchsummary.summary(net, input_size=(c,h,w))
    print(net(x).shape)