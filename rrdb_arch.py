import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scale_init(module, scale):
    # ESRGAN-style init: kaiming, then shrink so deep residual stacks start near identity
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()


class DenseBlock(nn.Module):
    def __init__(self, nf, res_scale=0.2):
        super(DenseBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nf*2, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nf*3, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=nf*4, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=nf*5, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        _scale_init(self, 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # residual dense block: x + scaled residual
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    def __init__(self, nf, res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.block1 = DenseBlock(nf, res_scale)
        self.block2 = DenseBlock(nf, res_scale)
        self.block3 = DenseBlock(nf, res_scale)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        # Same as the official ESRGAN RRDB: out * 0.2 + x
        return x + out * self.res_scale


class SRN(nn.Module):
    """RRDB super-resolution net.

    - Features are upsampled with conv + PixelShuffle(2) stages, so every HR
      sub-pixel position gets its own learned channels (nearest upsampling
      could not produce sub-pixel detail).
    - The network predicts a residual on top of a bicubic upscale of the input,
      so it starts at the bicubic baseline and only has to learn the missing detail.
    """

    def __init__(self, num_channels=1, nf=64, scale_factor=4, num_blocks=12):
        super(SRN, self).__init__()
        assert scale_factor >= 2 and (scale_factor & (scale_factor - 1)) == 0, (
            f"scale_factor must be a power of 2 (got {scale_factor})"
        )
        self.scale_factor = scale_factor

        self.first_conv = nn.Conv2d(num_channels, nf, kernel_size=3, stride=1, padding=1)
        self.RRDB_blocks = nn.ModuleList([RRDB(nf) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        up = []
        for _ in range(int(math.log2(scale_factor))):
            up += [
                nn.Conv2d(nf, nf * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*up)

        self.high_res_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.last_conv = nn.Conv2d(nf, num_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Zero the last layer so the untrained model outputs exactly the bicubic upscale.
        nn.init.zeros_(self.last_conv.weight)
        nn.init.zeros_(self.last_conv.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)

        feature_map = self.first_conv(x)
        fmap = feature_map
        for rrdb in self.RRDB_blocks:
            fmap = rrdb(fmap)
        feature_map = feature_map + self.trunk_conv(fmap)

        feature_map = self.upsample(feature_map)
        feature_map = self.lrelu(self.high_res_conv(feature_map))
        residual = self.last_conv(feature_map)

        out = base + residual
        # No clamp while training (clamping zeroes gradients for out-of-range pixels);
        # clamp at eval so metrics and saved images are in [0, 1].
        if not self.training:
            out = out.clamp(0.0, 1.0)
        return out
