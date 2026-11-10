import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Runs the ResNet up to and including `up_to`.

    up_to="avgpool" (old default, kept for edgemap_finetune.py) pools everything
    into one global vector, which has no spatial information and is a weak
    perceptual signal. For SR, use an intermediate block such as "block7"
    (end of layer 2: 512 channels at 1/8 resolution).
    """

    def __init__(self, resnet, up_to="avgpool"):
        super().__init__()

        names = ["conv1", "bn1", "relu", "max_pool"] + [f"block{i}" for i in range(1, 17)] + ["avgpool"]
        assert up_to in names, f"up_to must be one of {names}"
        layers = []
        for name in names:
            layers.append(getattr(resnet, name))
            if name == up_to:
                break
        self.features = nn.Sequential(*layers)

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
    if x.shape[-2:] == (256, 256):
        return x
    return F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)

def normalize_tensor(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

def prep_for_resnet(x):
    x = to_rgb(x)
    x = resize_for_resnet(x)
    x = normalize_tensor(x)
    return x