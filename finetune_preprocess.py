import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class ImagePairDataset(Dataset):
    """
    Loads matching low-res / high-res image pairs from a folder structure like:

        root_dir/
            low_res/
                img001.png
                img002.png
                ...
            high_res/
                img001.png
                img002.png
                ...

    IMPORTANT: hr_crop_size is not a free choice -- it must match your
    model's fixed output size (SRN's nn.Upsample(size=output_size)).
    We fine-tune on 512x512 HR crops, so hr_crop_size=512 here must
    match `SRN(1, 64, 512)` in run_model.py. The low-res crop size is
    derived from that: hr_crop_size / scale_factor.
    """

    def __init__(self, root_dir, hr_crop_size=512, scale_factor=4, is_train=True):
        self.input_dir = os.path.join(root_dir, "DIV_2K_LR/X4")
        self.target_dir = os.path.join(root_dir, "DIV_2K_HR")

        self.inputs = sorted(os.listdir(self.input_dir))
        self.targets = sorted(os.listdir(self.target_dir))

        assert len(self.inputs) == len(self.targets), (
            f"Found {len(self.inputs)} low-res images but "
            f"{len(self.targets)} high-res images -- they should match 1:1."
        )
        assert hr_crop_size % scale_factor == 0, (
            f"hr_crop_size ({hr_crop_size}) must divide evenly by "
            f"scale_factor ({scale_factor})."
        )

        self.hr_crop_size = hr_crop_size
        self.scale_factor = scale_factor
        self.lr_crop_size = hr_crop_size // scale_factor  # e.g. 512 // 4 = 128

        # is_train=True  -> crop a random region each time (adds a bit of
        #                    variety, which helps if you don't have many
        #                    training images).
        # is_train=False -> always crop the exact center (so your test-set
        #                    numbers are reproducible from run to run).
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)

    def _paired_crop(self, low_res_img, high_res_img):
        """
        Crop the same region out of both images. `low_res_img` and
        `high_res_img` are PIL Images of the same scene, where
        high_res_img is `scale_factor` times bigger in each dimension.
        """
        lr_w, lr_h = low_res_img.size
        crop = self.lr_crop_size

        # If the low-res image is smaller than the crop size we need,
        # there's nothing to crop -- so first scale BOTH images up
        # (by the same factor) until the low-res side is big enough.
        # This keeps the low-res/high-res relationship intact; it just
        # makes the whole pair bigger before we cut our fixed-size
        # window out of it.
        if lr_w < crop or lr_h < crop:
            up_factor = max(crop / lr_w, crop / lr_h)
            new_lr_w, new_lr_h = int(lr_w * up_factor + 0.5), int(lr_h * up_factor + 0.5)
            low_res_img = low_res_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            high_res_img = high_res_img.resize(
                (new_lr_w * self.scale_factor, new_lr_h * self.scale_factor), Image.BICUBIC
            )
            lr_w, lr_h = new_lr_w, new_lr_h

        # Choose the top-left corner of the crop, in low-res coordinates.
        max_x = lr_w - crop
        max_y = lr_h - crop
        if self.is_train:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            x = max_x // 2
            y = max_y // 2

        low_res_crop = TF.crop(low_res_img, top=y, left=x, height=crop, width=crop)

        # Same region in the high-res image: just multiply everything
        # by scale_factor, since it's scale_factor times bigger overall.
        s = self.scale_factor
        high_res_crop = TF.crop(
            high_res_img, top=y * s, left=x * s, height=crop * s, width=crop * s
        )

        return low_res_crop, high_res_crop

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.inputs[idx])
        target_path = os.path.join(self.target_dir, self.targets[idx])

        # convert imgs to grayscale
        low_res_img = Image.open(input_path).convert("L")
        high_res_img = Image.open(target_path).convert("L")

        low_res_img, high_res_img = self._paired_crop(low_res_img, high_res_img)

        low_res_tensor = self.to_tensor(low_res_img)
        high_res_tensor = self.to_tensor(high_res_img)

        return low_res_tensor, high_res_tensor