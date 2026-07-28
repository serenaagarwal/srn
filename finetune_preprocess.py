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

    The two folders must contain the same number of files, sorted into
    matching order (low_res/img001.png pairs with high_res/img001.png).

    Since your images vary in size, every pair gets cropped down to a
    fixed size before being returned -- lr_crop_size for the low-res
    image, and lr_crop_size * scale_factor for the high-res image. The
    crop is "paired": the same region of the scene is cut out of both
    images, just at their respective resolutions, so they still line up
    pixel-for-pixel.
    """

    def __init__(self, root_dir, lr_crop_size=128, scale_factor=4, is_train=True):
        self.input_dir = os.path.join(root_dir, "low_res")
        self.target_dir = os.path.join(root_dir, "high_res")

        self.inputs = sorted(os.listdir(self.input_dir))
        self.targets = sorted(os.listdir(self.target_dir))

        assert len(self.inputs) == len(self.targets), (
            f"Found {len(self.inputs)} low-res images but "
            f"{len(self.targets)} high-res images -- they should match 1:1."
        )

        self.lr_crop_size = lr_crop_size
        self.scale_factor = scale_factor
        self.hr_crop_size = lr_crop_size * scale_factor

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

        # If the low-res image is smaller than the crop size we want,
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

        # "L" = single-channel grayscale. Change to "RGB" below (both
        # places) if your lab images are actually in color.
        low_res_img = Image.open(input_path).convert("L")
        high_res_img = Image.open(target_path).convert("L")

        low_res_img, high_res_img = self._paired_crop(low_res_img, high_res_img)

        low_res_tensor = self.to_tensor(low_res_img)
        high_res_tensor = self.to_tensor(high_res_img)

        return low_res_tensor, high_res_tensor