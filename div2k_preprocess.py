import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(folder):
    # sorted image files only (skips .DS_Store, Thumbs.db, etc.)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


class ResizePairDataset(Dataset):
    """Same behavior as the old preprocess.py (used for kaggle): the LR image is
    resized to lr_size x lr_size, the HR image is used as-is. No cropping or
    augmentation, and no size check. Only suitable when every HR image has the
    same size (e.g. all 256x256) so they can be batched."""

    def __init__(self, lr_dir, hr_dir, lr_size=128):
        self.input_dir = lr_dir
        self.target_dir = hr_dir
        self.inputs = _list_images(lr_dir)
        self.targets = _list_images(hr_dir)
        assert len(self.inputs) == len(self.targets), (
            f"Found {len(self.inputs)} low-res images in {lr_dir} but "
            f"{len(self.targets)} high-res images in {hr_dir} -- they should match 1:1."
        )
        self.lr_transform = T.Compose([T.Resize((lr_size, lr_size)), T.ToTensor()])
        self.hr_transform = T.ToTensor()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.input_dir, self.inputs[idx])).convert("L")
        mask = Image.open(os.path.join(self.target_dir, self.targets[idx])).convert("L")
        return self.lr_transform(img), self.hr_transform(mask)


class ImagePairDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, hr_crop_size=256, scale_factor=4, is_train=True, downscale_lr=False):
        self.input_dir = lr_dir
        self.target_dir = hr_dir

        self.inputs = _list_images(self.input_dir)
        self.targets = _list_images(self.target_dir)

        assert len(self.inputs) == len(self.targets), (
            f"Found {len(self.inputs)} low-res images in {lr_dir} but "
            f"{len(self.targets)} high-res images in {hr_dir} -- they should match 1:1."
        )
        assert hr_crop_size % scale_factor == 0, (
            f"hr_crop_size ({hr_crop_size}) must divide evenly by "
            f"scale_factor ({scale_factor})."
        )

        self.hr_crop_size = hr_crop_size
        self.scale_factor = scale_factor
        self.lr_crop_size = hr_crop_size // scale_factor  # e.g., 256 // 4 = 64
        self.is_train = is_train
        # For datasets whose LR is stored at full size but is really scale_factor x lower
        # resolution (e.g. kaggle): shrink LR to HR / scale_factor before anything else.
        self.downscale_lr = downscale_lr
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)

    def _paired_crop(self, low_res_img, high_res_img):
        lr_w, lr_h = low_res_img.size
        crop = self.lr_crop_size

        # Ensure LR image is big enough for crop
        if lr_w < crop or lr_h < crop:
            up_factor = max(crop / lr_w, crop / lr_h)
            new_lr_w, new_lr_h = int(lr_w * up_factor + 0.5), int(lr_h * up_factor + 0.5)
            low_res_img = low_res_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            high_res_img = high_res_img.resize(
                (new_lr_w * self.scale_factor, new_lr_h * self.scale_factor), Image.BICUBIC
            )
            lr_w, lr_h = new_lr_w, new_lr_h

        max_x = lr_w - crop
        max_y = lr_h - crop

        if self.is_train:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            x = max_x // 2
            y = max_y // 2

        low_res_crop = TF.crop(low_res_img, top=y, left=x, height=crop, width=crop)

        s = self.scale_factor
        # HR crop coordinates must be scaled by scale_factor
        high_res_crop = TF.crop(
            high_res_img, top=y * s, left=x * s, height=crop * s, width=crop * s
        )

        return low_res_crop, high_res_crop

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.inputs[idx])
        target_path = os.path.join(self.target_dir, self.targets[idx])

        low_res_img = Image.open(input_path).convert("L")
        high_res_img = Image.open(target_path).convert("L")

        if self.downscale_lr:
            assert low_res_img.size == high_res_img.size, (
                f"downscale_lr expects LR and HR to be the same size, but "
                f"{self.inputs[idx]} is LR {low_res_img.size}, HR {high_res_img.size}"
            )
            low_res_img = low_res_img.resize(
                (high_res_img.size[0] // self.scale_factor, high_res_img.size[1] // self.scale_factor),
                Image.BICUBIC,
            )

        # HR should be scale_factor x LR. Many datasets make LR by rounding DOWN
        # (e.g. HR 510 -> LR 127, and 127*4 = 508), leaving HR a few pixels larger.
        # Trim that extra strip off the right/bottom edge; anything else is an error.
        s = self.scale_factor
        exp_w, exp_h = low_res_img.size[0] * s, low_res_img.size[1] * s
        hr_w, hr_h = high_res_img.size
        if (hr_w, hr_h) != (exp_w, exp_h):
            assert exp_w <= hr_w < exp_w + s and exp_h <= hr_h < exp_h + s, (
                f"Size mismatch for {self.inputs[idx]} / {self.targets[idx]}: "
                f"LR {low_res_img.size}, HR {high_res_img.size}, scale {s}"
            )
            high_res_img = high_res_img.crop((0, 0, exp_w, exp_h))

        low_res_img, high_res_img = self._paired_crop(low_res_img, high_res_img)
        if self.is_train:
            # Synchronized random flipping/rotation
            if random.random() > 0.5:
                low_res_img = TF.hflip(low_res_img)
                high_res_img = TF.hflip(high_res_img)
            if random.random() > 0.5:
                low_res_img = TF.vflip(low_res_img)
                high_res_img = TF.vflip(high_res_img)

            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                low_res_img = TF.rotate(low_res_img, angle)
                high_res_img = TF.rotate(high_res_img, angle)
        low_res_tensor = self.to_tensor(low_res_img)
        high_res_tensor = self.to_tensor(high_res_img)

        if self.is_train:
            low_res_tensor = low_res_tensor + 0.01 * torch.randn_like(low_res_tensor)
            low_res_tensor = torch.clamp(low_res_tensor, 0.0, 1.0)

        return low_res_tensor, high_res_tensor