import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, nf):
        super(DenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nf*2, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nf*3, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=nf*4, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=nf*5, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class RRDB(nn.Module):
    def __init__(self, nf):
        super(RRDB, self).__init__()
        self.res_scale = 0.2
        self.block1 = DenseBlock(nf)
        self.block2 = DenseBlock(nf)
        self.block3 = DenseBlock(nf)
    
    def forward(self, x):
        out = x + self.block1(x) * self.res_scale
        out = out + self.block2(out) * self.res_scale
        out = out + self.block3(out) * self.res_scale
        
        return x + out*self.res_scale

class SRN(nn.Module):
    def __init__(self, num_channels, nf, output_size):
        super(SRN, self).__init__()
  
        self.first_conv = nn.Conv2d(in_channels=num_channels, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.RRDB_blocks = nn.ModuleList([RRDB(64) for _ in range(12)]) #making a deeper model will help
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=output_size)
        #trying out another upsample method
        # self.up1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        # self.pixel_shuffle1 = nn.PixelShuffle(2)
        # self.up2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        # self.pixel_shuffle2 = nn.PixelShuffle(2)
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
        # feature_map = self.up1(feature_map)
        # feature_map = self.pixel_shuffle1(feature_map)
        # feature_map = self.lrelu(feature_map)

        # feature_map = self.up2(feature_map)
        # feature_map = self.pixel_shuffle2(feature_map)
        # feature_map = self.lrelu(feature_map)

        #high_res_features = self.lrelu(self.high_res_conv(feature_map))
        #output = self.last_conv(high_res_features)
        
        return torch.sigmoid(output)

        

        
