import torch
import torch.nn as nn
import numpy as np

class Block(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):

        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    

class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #layer 1 -> 3 blocks
        self.block1 = Block(self.in_channels, out_channels=64, identity_downsample = self.identity_downsample(64, 64, stride=1), stride=1)
        self.block2 = Block(256, out_channels=64, identity_downsample = None, stride=1)
        self.block3 = Block(256, out_channels=64, identity_downsample = None, stride=1)

        #layer 2 -> 4 blocks
        self.block4 = Block(256, 128, identity_downsample = self.identity_downsample(256, 128, stride=2), stride=2)
        self.block5 = Block(512, 128, identity_downsample = None, stride = 1)
        self.block6 = Block(512, 128, identity_downsample = None, stride = 1)
        self.block7 = Block(512, 128, identity_downsample = None, stride = 1)

        #layer 3 -> 6 blocks
        self.block8 = Block(512, 256, identity_downsample = self.identity_downsample(512, 256, stride=2), stride=2)
        self.block9 = Block(1024, 256, identity_downsample = None, stride = 1)
        self.block10 = Block(1024, 256, identity_downsample = None, stride = 1)
        self.block11 = Block(1024, 256, identity_downsample = None, stride = 1)
        self.block12 = Block(1024, 256, identity_downsample = None, stride = 1)
        self.block13 = Block(1024, 256, identity_downsample = None, stride = 1)

        #layer 4 -> 3 blocks
        self.block14 = Block(1024, 512, identity_downsample = self.identity_downsample(1024, 512, stride=2), stride=2)
        self.block15 = Block(2048, 512, identity_downsample = None, stride = 1)
        self.block16 = Block(2048, 512, identity_downsample = None, stride = 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes) #fully connected layer

    def identity_downsample(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels*Block.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels*Block.expansion)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    