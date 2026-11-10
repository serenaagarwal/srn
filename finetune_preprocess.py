import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Fixed validation crop positions (fractions of the free space in x, y):
# center, then the four quadrant centers. Same as div2k_preprocess.py.
VAL_CROP_POSITIONS = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]


def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def load_intensity(path):
    """Load a microscopy image as single-channel intensity (8-bit PIL 'L').

    The borton PNGs are palette images with a green LUT: value i is stored as
    color (0, i, 0). PIL's .convert("L") turns that into luminance 0.587*i, so
    the brightest pixel (255) would load as ~150 and the dim range gets
    squashed. Taking the max over R, G, B recovers i exactly for a green (or
    any single-color / gray) LUT, and is a no-op for ordinary grayscale files.
    """
    img = Image.open(path)
    if img.mode in ("L", "I;16", "I", "F"):
        return img.convert("L")
    rgb = np.asarray(img.convert("RGB"))
    return Image.fromarray(rgb.max(axis=2).astype(np.uint8), mode="L")


class ImagePairDataset(Dataset):
    """
    Loads LR / HR microscopy pairs from:

        root_dir/
            LR/   (ImageJ bicubic downscale of HR, ~HR/4)
            HR/

    Files are paired by sorted order. (In borton_data/train, 454 LR filenames
    differ from their HR names, e.g. "C=0-20000" vs "C=00000", but every pair
    was checked by content and matches in sorted order. Renaming them to match
    would be safer.)

    hr_crop_size=256 (LR 64) fits every image, including the 93x93 LR test
    images; the old 512 crop forced those to be bicubic-UPSCALED first, so the
    "ground truth" was an interpolated image. The model (rrdb_arch.SRN) works at
    any size, and 256 is divisible by 16 as the segmentation U-Net requires.

    HR is not exactly 4x LR (e.g. 597 vs 149, 543 vs 136), because ImageJ maps
    the full HR width onto the LR width. HR crop offsets therefore use the true
    per-image HR/LR ratio instead of a flat 4, keeping LR and HR aligned to
    within ~0.5 HR pixel.

    is_train=True  -> crops_per_image random crops per image, plus synchronized
                      flips / 90-degree rotations.
    is_train=False -> up to 5 fixed crops (center + quadrants), identical every
                      epoch, so validation numbers are reproducible.
    """

    def __init__(self, root_dir, hr_crop_size=256, scale_factor=4, is_train=True, crops_per_image=1,
                 lr_subdir="LR", hr_subdir="HR"):
        self.input_dir = os.path.join(root_dir, lr_subdir)
        self.target_dir = os.path.join(root_dir, hr_subdir)

        self.inputs = _list_images(self.input_dir)
        self.targets = _list_images(self.target_dir)

        assert len(self.inputs) == len(self.targets), (
            f"Found {len(self.inputs)} low-res images in {self.input_dir} but "
            f"{len(self.targets)} high-res images in {self.target_dir} -- they should match 1:1."
        )
        assert hr_crop_size % scale_factor == 0, (
            f"hr_crop_size ({hr_crop_size}) must divide evenly by scale_factor ({scale_factor})."
        )
        if not is_train:
            assert 1 <= crops_per_image <= len(VAL_CROP_POSITIONS), (
                f"validation supports 1-{len(VAL_CROP_POSITIONS)} crops per image"
            )

        self.hr_crop_size = hr_crop_size
        self.scale_factor = scale_factor
        self.lr_crop_size = hr_crop_size // scale_factor
        self.is_train = is_train
        self.crops_per_image = crops_per_image
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs) * self.crops_per_image

    def _paired_crop(self, low_res_img, high_res_img, crop_id=0):
        lr_w, lr_h = low_res_img.size
        hr_w, hr_h = high_res_img.size
        crop, hr_crop = self.lr_crop_size, self.hr_crop_size

        assert lr_w >= crop and lr_h >= crop, (
            f"LR image {low_res_img.size} is smaller than the {crop}px LR crop; lower hr_crop_size"
        )
        # sanity check that this really is a ~scale_factor pair
        rx, ry = hr_w / lr_w, hr_h / lr_h
        assert abs(rx - self.scale_factor) < 0.1 and abs(ry - self.scale_factor) < 0.1, (
            f"LR {low_res_img.size} / HR {high_res_img.size} is not a x{self.scale_factor} pair"
        )

        max_x, max_y = lr_w - crop, lr_h - crop
        if self.is_train:
            x, y = random.randint(0, max_x), random.randint(0, max_y)
        else:
            fx, fy = VAL_CROP_POSITIONS[crop_id]
            x, y = int(round(max_x * fx)), int(round(max_y * fy))

        low_res_crop = TF.crop(low_res_img, top=y, left=x, height=crop, width=crop)

        # same region in HR, using the true HR/LR ratio of this image
        hx = min(int(round(x * rx)), hr_w - hr_crop)
        hy = min(int(round(y * ry)), hr_h - hr_crop)
        high_res_crop = TF.crop(high_res_img, top=hy, left=hx, height=hr_crop, width=hr_crop)

        return low_res_crop, high_res_crop

    def __getitem__(self, idx):
        idx, crop_id = idx % len(self.inputs), idx // len(self.inputs)
        low_res_img = load_intensity(os.path.join(self.input_dir, self.inputs[idx]))
        high_res_img = load_intensity(os.path.join(self.target_dir, self.targets[idx]))

        low_res_img, high_res_img = self._paired_crop(low_res_img, high_res_img, crop_id)

        if self.is_train:
            if random.random() > 0.5:
                low_res_img, high_res_img = TF.hflip(low_res_img), TF.hflip(high_res_img)
            if random.random() > 0.5:
                low_res_img, high_res_img = TF.vflip(low_res_img), TF.vflip(high_res_img)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                low_res_img, high_res_img = TF.rotate(low_res_img, angle), TF.rotate(high_res_img, angle)

        return self.to_tensor(low_res_img), self.to_tensor(high_res_img)
