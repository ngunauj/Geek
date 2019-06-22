# coding=utf-8
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    '''
    resnet block
    '''

    def __init__(self, ch_in, ch_out, stride=1):
        '''
        :param ch_in:
        :param ch_out:
        '''
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=3,
            stride=stride,
            padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        '''
        :param x:
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.blk1 = ResBlock(64, 64, 1)
        self.blk2 = ResBlock(64, 64, 1)
        self.blk3 = ResBlock(64, 128, 2)
        self.blk4 = ResBlock(128, 128, 1)
        self.blk5 = ResBlock(128, 256, 2)
        self.blk6 = ResBlock(256, 256, 1)
        self.blk7 = ResBlock(256, 512, 2)
        self.blk8 = ResBlock(512, 512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18():
    return ResNet(ResBlock)
