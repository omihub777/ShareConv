import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class SConv2d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SConv2d, self).__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = torch.sum(x, dim=1, keepdim=True) # sum across channels
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class SConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(SConvBlock, self).__init__()
        self.conv1 = SConv2d(out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, stride=1, padding=p, bias=bias)

        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, bias=False)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))

        return out + self.skip(x)

class SPreActBlock(nn.Module):
    """PreActBlock using Shared Convolutin"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(SPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = SConv2d(out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = SConv2d(out_c, kernel_size=k, stride=1, padding=p, bias=bias)

        if s!=1 or in_c!=out_c:
            self.skip = SConv2d(out_c, kernel_size=1,stride=s, bias=False)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))

        return out + self.skip(x)

if __name__ == "__main__":
    b,c,h,w = 4, 16, 32, 32
    x = torch.randn(b, c, h, w)
    out_c, s = 64, 2
    blc = SPreActBlock(c, out_c, s=s)
    # blc = PreActBlock(c, out_c, s=s)
    out = blc(x)
    torchsummary.summary(blc, (c,h,w))
    print(out.shape)
