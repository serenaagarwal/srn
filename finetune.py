"""Fine-tune the pretrained SRN on the borton microglia data (LR image only, no edge map).

Same structure as edgemap_finetune.py, minus the edge-map input. Losses:
    content      = alpha * (1 - MS-SSIM) + (1 - alpha) * L1
    perceptual   = ResNet feature loss (weight lambda_resnet; 0 disables it)
    segmentation = BCE between U-Net(SR) and U-Net(HR) (weight lambda_seg)

Validation reports, for both SR and a plain bicubic upscale of LR:
    MS-SSIM, PSNR,
    Dice of U-Net masks vs the U-Net mask of HR  (the network used in the loss)
    Dice of Otsu-threshold masks vs Otsu of HR   (independent of the loss)

Usage:
    python finetune.py --checkpoint saved_models/srn_flickr2_best_epochXX_msssimX.pth
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM
import matplotlib.pyplot as plt
import wandb

from finetune_preprocess import ImagePairDataset
from rrdb_arch import SRN
from resnet import ResNet
from seg_loss_unet import U_Net
from perceptual_loss import FeatureExtractor, PerceptualLoss, prep_for_resnet
from run_model import msssim_per_image, psnr_per_image, save_checkpoint

ENTITY = "serena-agarwal-brown-university"
PROJECT = "srn-model"


def load_frozen(model, path, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def otsu_masks(x, bins=256):
    """Per-image Otsu threshold -> binary mask. x: (B,1,H,W) in [0,1]."""
    masks = []
    for img in x:
        v = img.flatten().clamp(0, 1)
        hist = torch.histc(v, bins=bins, min=0.0, max=1.0)
        p = hist / hist.sum()
        w = torch.cumsum(p, 0)
        mu = torch.cumsum(p * torch.arange(bins, device=x.device, dtype=p.dtype), 0)
        between = (mu[-1] * w - mu) ** 2 / (w * (1 - w)).clamp_min(1e-12)
        t = (torch.argmax(between).float() + 0.5) / bins
        masks.append(img > t)
    return torch.stack(masks)


def dice_per_image(a, b, eps=1e-6):
    a, b = a.flatten(1).float(), b.flatten(1).float()
    inter = (a * b).sum(1)
    total = a.sum(1) + b.sum(1)
    # both empty -> perfect agreement
    return torch.where(total > 0, 2 * inter / (total + eps), torch.ones_like(total))


def save_image_comparison(lr, sr, hr, sr_mask, hr_mask, epoch, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    panels = [(lr, "Low-Res Input"), (sr, "SRN Output"), (hr, "High-Res Ground Truth"),
              (sr_mask, "U-Net mask (SR)"), (hr_mask, "U-Net mask (HR)")]
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (img, title) in zip(axs, panels):
        ax.imshow(img.squeeze().detach().float().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    path = f"{save_dir}/epoch{epoch}_sample{idx}.png"
    plt.savefig(path)
    plt.close()
    wandb.log({f"sample{idx}": wandb.Image(path), "epoch": epoch})


def main(args):
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"borton-finetune-{args.tag}",
        group="borton",
        config={
            "tag": args.tag,
            "finetune_from": args.checkpoint,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hr_crop_size": args.hr_crop_size,
            "train_crops_per_image": args.train_crops_per_image,
            "val_crops_per_image": args.val_crops_per_image,
            "alpha_content": args.alpha_content,  # 0.84, from SFSRM
            "lambda_resnet": args.lambda_resnet,
            "lambda_seg": args.lambda_seg,
            "min_lr_ratio": 0.01,
        },
    )
    config = run.config
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    train_dataset = ImagePairDataset(f"{args.data_root}/train", hr_crop_size=config.hr_crop_size, is_train=True,
                                     crops_per_image=config.train_crops_per_image)
    test_dataset = ImagePairDataset(f"{args.data_root}/test", hr_crop_size=config.hr_crop_size, is_train=False,
                                    crops_per_image=config.val_crops_per_image)
    print(f"Dataset created! train={len(train_dataset)} val={len(test_dataset)} crops", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}!", flush=True)

    # SR model: same architecture as run_model.py, initialized from the pretrained checkpoint.
    # Only the weights are reused; epoch counter, best metric and optimizer state start fresh,
    # since they belong to a different dataset.
    model = SRN(num_channels=1, nf=64, scale_factor=4).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded pretrained weights from {args.checkpoint} "
          f"(pretraining epoch {ck.get('epoch')}, MS-SSIM {ck.get('best_metric', float('nan')):.6f})", flush=True)

    # Frozen perceptual-loss network
    resnet = load_frozen(ResNet(3, 23), "saved_models/new_resnet.pth", device)
    perceptual_loss = PerceptualLoss(FeatureExtractor(resnet).to(device)).to(device)

    # Frozen segmentation network (U-Net trained on ilastik-labelled microglia)
    u_net = load_frozen(U_Net(64), "saved_models/new_unet.pth", device)

    l1_criterion = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    ms_ssim_fn = MS_SSIM(data_range=1.0, size_average=False, channel=1)  # use via msssim_per_image
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader), eta_min=config.learning_rate * config.min_lr_ratio
    )
    print("Initialized loss and optimizer!", flush=True)

    best_ms_ssim = 0.0
    num_images_to_save = 5

    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "content loss": 0.0, "resnet loss": 0.0, "seg loss": 0.0}
        total_grad_norm, num_batches = 0.0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            ms_ssim_loss = 1.0 - msssim_per_image(ms_ssim_fn, outputs, masks).mean()
            l1_loss = l1_criterion(outputs, masks)
            content_loss = config.alpha_content * ms_ssim_loss + (1.0 - config.alpha_content) * l1_loss

            if config.lambda_resnet > 0:
                resnet_loss = perceptual_loss(prep_for_resnet(outputs), prep_for_resnet(masks))
            else:
                resnet_loss = torch.zeros((), device=device)

            # Segmentation loss: U-Net(SR) should match U-Net(HR).
            # The U-Net outputs logits, so the HR target is turned into probabilities with a
            # sigmoid -- BCEWithLogitsLoss needs targets in [0, 1].
            sr_seg_logits = u_net(outputs)
            with torch.no_grad():
                gt_seg_prob = torch.sigmoid(u_net(masks))
            seg_loss = bce_loss(sr_seg_logits, gt_seg_prob)

            total = content_loss + config.lambda_resnet * resnet_loss + config.lambda_seg * seg_loss

            optimizer.zero_grad()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            n = images.size(0)
            sums["loss"] += total.item() * n
            sums["content loss"] += content_loss.item() * n
            sums["resnet loss"] += resnet_loss.item() * n
            sums["seg loss"] += seg_loss.item() * n
            total_grad_norm += grad_norm.item()
            num_batches += 1

        train_log = {k: v / len(train_dataset) for k, v in sums.items()}
        train_log.update({"epoch": epoch, "total norm": total_grad_norm / num_batches,
                          "lr": scheduler.get_last_lr()[0]})
        wandb.log(train_log)
        print(f"Epoch {epoch}/{config.epochs} | Train loss {train_log['loss']:.6f} "
              f"(content {train_log['content loss']:.6f}, seg {train_log['seg loss']:.6f})", flush=True)

        # ---------------- validation ----------------
        model.eval()
        keys = ["msssim", "psnr", "dice_unet", "dice_otsu"]
        val = {f"{m}/{src}": 0.0 for m in keys for src in ("sr", "bicubic")}
        images_saved = 0

        with torch.no_grad():
            for val_images, val_masks in test_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                sr = model(val_images)
                bic = F.interpolate(val_images, scale_factor=4, mode="bicubic",
                                    align_corners=False).clamp(0.0, 1.0)

                hr_unet = torch.sigmoid(u_net(val_masks)) > 0.5
                hr_otsu = otsu_masks(val_masks)

                for name, pred in (("sr", sr), ("bicubic", bic)):
                    val[f"msssim/{name}"] += msssim_per_image(ms_ssim_fn, pred, val_masks).sum().item()
                    val[f"psnr/{name}"] += psnr_per_image(pred, val_masks).sum().item()
                    pred_unet = torch.sigmoid(u_net(pred)) > 0.5
                    val[f"dice_unet/{name}"] += dice_per_image(pred_unet, hr_unet).sum().item()
                    val[f"dice_otsu/{name}"] += dice_per_image(otsu_masks(pred), hr_otsu).sum().item()

                if images_saved < num_images_to_save:
                    sr_unet = torch.sigmoid(u_net(sr)) > 0.5
                    for j in range(min(num_images_to_save - images_saved, val_images.size(0))):
                        save_image_comparison(val_images[j], sr[j], val_masks[j], sr_unet[j], hr_unet[j],
                                              epoch=epoch, idx=images_saved,
                                              save_dir=f"visual_outputs/finetune_{args.tag}")
                        images_saved += 1

        val = {k: v / len(test_dataset) for k, v in val.items()}
        val["epoch"] = epoch
        wandb.log(val)
        print(f"Epoch {epoch}/{config.epochs} | "
              f"MS-SSIM {val['msssim/sr']:.6f} (bicubic {val['msssim/bicubic']:.6f}) | "
              f"PSNR {val['psnr/sr']:.3f} (bicubic {val['psnr/bicubic']:.3f}) | "
              f"Dice U-Net {val['dice_unet/sr']:.4f} (bicubic {val['dice_unet/bicubic']:.4f}) | "
              f"Dice Otsu {val['dice_otsu/sr']:.4f} (bicubic {val['dice_otsu/bicubic']:.4f})", flush=True)

        if val["msssim/sr"] > best_ms_ssim:
            best_ms_ssim = val["msssim/sr"]
            save_checkpoint(
                path=f"saved_models/srn_finetuned_{args.tag}_best_epoch{epoch}_msssim{best_ms_ssim:.6f}.pth",
                model=model, optimizer=optimizer, epoch=epoch, best_metric=best_ms_ssim, config=run.config,
            )
            print(f"New best MS-SSIM: {best_ms_ssim:.6f} - Model saved!", flush=True)

    print(f"\nFine-tuning complete! Best MS-SSIM achieved: {best_ms_ssim:.6f}", flush=True)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="saved_models/srn_flickr2_best_epoch14_msssim0.943895.pth",
                        help="pretrained SRN checkpoint from run_model.py (rrdb_arch.SRN)")
    parser.add_argument("--data_root", default="borton_data", help="folder with train/ and test/ (each with LR/, HR/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hr_crop_size", type=int, default=256)
    parser.add_argument("--train_crops_per_image", type=int, default=8)
    parser.add_argument("--val_crops_per_image", type=int, default=5)
    parser.add_argument("--alpha_content", type=float, default=0.84)
    parser.add_argument("--lambda_resnet", type=float, default=0.01, help="0 disables the perceptual loss")
    parser.add_argument("--lambda_seg", type=float, default=0.01)
    parser.add_argument("--tag", default=None,
                        help="label for this run's wandb name, checkpoints and sample images "
                             "(default: seg<lambda_seg>_res<lambda_resnet>)")
    args = parser.parse_args()
    if args.tag is None:
        args.tag = f"seg{args.lambda_seg:g}_res{args.lambda_resnet:g}"
    main(args)
