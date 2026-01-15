import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, nf):
        super(DenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.lrelu(x)
        return x

class RRDB(nn.Module):
    def __init__(self, nf):
        super(RRDB, self).__init__()
        self.block1 = DenseBlock(nf)
        self.block2 = DenseBlock(nf)
        self.block3 = DenseBlock(nf)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = x + out
        return out

class SRN(nn.Module):
    def __init__(self, num_channels, nf, output_size):
        super(SRN, self).__init__()
        nf = 64
        self.first_conv = nn.Conv2d(in_channels=num_channels, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.RRDB_blocks = nn.ModuleList([RRDB(64) for _ in range(8)])
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=output_size)
        self.high_res_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.last_conv = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        feature_map = self.first_conv(x)
        fmap = feature_map
        for rrdb in self.RRDB_blocks:
            fmap = rrdb(fmap)
        trunk = self.trunk_conv(fmap)
        feature_map = feature_map + trunk
        feature_map = self.upsample(feature_map)
        activated = self.lrelu(feature_map)
        output = self.last_conv(activated)
        
        return output

        

        
