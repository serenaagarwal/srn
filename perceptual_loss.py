import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.max_pool,
            resnet.block1,
            resnet.block2,
            resnet.block3,
            resnet.block4,
            resnet.block5,
            resnet.block6,
            resnet.block7,
            resnet.block8,
            resnet.block9,
            resnet.block10,
            resnet.block11,
            resnet.block12,
            resnet.block13,
            resnet.block14, 
            resnet.block15,
            resnet.block16,
            resnet.avgpool
        )

        for p in self.features.parameters():
            p.requires_grad = False
        

    def forward(self, x):
        return self.features(x)

class PerceptualLoss(nn.Module):
     def __init__(self, feature_extractor):
        super().__init__()
        self.feat = feature_extractor
        self.loss = nn.L1Loss()

     def forward(self, sr, gt):
        #with torch.no_grad():
        feat_sr = self.feat(sr)
        feat_gt = self.feat(gt).detach()
        
        return self.loss(feat_sr, feat_gt)
     
def to_rgb(x):
   return x.repeat(1, 3, 1, 1)

def resize_for_resnet(x):
    return F.interpolate(x, (256, 256))

def normalize_tensor(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

def prep_for_resnet(x):
    x = to_rgb(x)
    x = resize_for_resnet(x)
    x = normalize_tensor(x)
    return x