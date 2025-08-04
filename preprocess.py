import os 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

#directory structure: 
# /train
#     /highres
#     /lowres
# /test
#     /highres
#     /lowres

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=True):
        
        self.transform = transform

        self.input_dir = os.path.join(root_dir, "low_res")
        self.mask_dir = os.path.join(root_dir, "high_res")
        self.inputs = sorted(os.listdir(self.input_dir))
        self.masks = sorted(os.listdir(self.mask_dir))


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.inputs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(input_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        transform_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        transform_mask = transforms.ToTensor()

        if (self.transform==True):
            img = transform_img(img)
            mask = transform_mask(mask)

        return img, mask



