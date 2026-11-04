import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x


class U_Net(nn.Module):
    def __init__(self, nf):
        super().__init__()


        #encoder!!

        self.conv1 = ConvBlock(in_channels=1, out_channels=nf)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(in_channels=nf, out_channels=nf*2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(in_channels=nf*2, out_channels=nf*4)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(in_channels=nf*4, out_channels=nf*8)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(in_channels=nf*8, out_channels=nf*16)

        #decoder!!
        self.upsample1 = nn.ConvTranspose2d(in_channels=nf*16, out_channels=nf*8, kernel_size=2, stride=2)
        self.upconv1 = ConvBlock(in_channels=nf*16, out_channels=nf*8)

        self.upsample2 = nn.ConvTranspose2d(in_channels=nf*8, out_channels=nf*4, kernel_size=2, stride=2)
        self.upconv2 = ConvBlock(in_channels=nf*8, out_channels=nf*4)

        self.upsample3 = nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=2, stride=2)
        self.upconv3 = ConvBlock(in_channels=nf*4, out_channels=nf*2)

        self.upsample4 = nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=2, stride=2)
        self.upconv4 = ConvBlock(in_channels=nf*2, out_channels=nf)
        
        self.output_conv = nn.Conv2d(nf, 1, kernel_size=1)
        

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1_pool = self.max_pool1(x1)

        x2 = self.conv2(x1_pool)
        x2_pool = self.max_pool2(x2)

        x3 = self.conv3(x2_pool)
        x3_pool = self.max_pool3(x3)

        x4 = self.conv4(x3_pool)
        x4_pool = self.max_pool4(x4)

        x_bottleneck = self.bottleneck(x4_pool)

        up1 = self.upsample1(x_bottleneck)
        up_cat1 = torch.cat([up1, x4], dim=1)
        upconv1 = self.upconv1(up_cat1)

        up2 = self.upsample2(upconv1)
        up_cat2 = torch.cat([up2, x3], dim=1)
        upconv2 = self.upconv2(up_cat2)

        up3 = self.upsample3(upconv2)
        up_cat3 = torch.cat([up3, x2], dim=1)
        upconv3 = self.upconv3(up_cat3)

        up4 = self.upsample4(upconv3)
        up_cat4 = torch.cat([up4, x1], dim=1)
        upconv4 = self.upconv4(up_cat4)

        output_tensor = self.output_conv(upconv4)

        return output_tensor