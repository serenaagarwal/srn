import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from div2k_preprocess import ImagePairDataset, ResizePairDataset
from rrdb_arch import SRN
from pytorch_msssim import MS_SSIM
from resnet import ResNet
from perceptual_loss import FeatureExtractor, PerceptualLoss, prep_for_resnet
import matplotlib.pyplot as plt
import wandb


def save_image_comparison(input_img, output_img, target_img, epoch, idx, dataset, save_dir="visual_outputs"):
    save_dir = os.path.join(save_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)

    input_img = input_img.squeeze().detach().cpu().numpy()
    output_img = output_img.squeeze().detach().cpu().numpy()
    target_img = target_img.squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_img, cmap='gray')
    axs[0].set_title("Low-Res Input")
    axs[0].axis('off')

    axs[1].imshow(output_img, cmap='gray')
    axs[1].set_title("SRN Output")
    axs[1].axis('off')

    axs[2].imshow(target_img, cmap='gray')
    axs[2].set_title("High-Res Ground Truth")
    axs[2].axis('off')

    plt.tight_layout()
    path = f"{save_dir}/epoch{epoch}_sample{idx}.png"
    plt.savefig(path)
    # include "epoch" so this log lines up with the other metrics
    wandb.log({f"sample{idx}": wandb.Image(path), "epoch": epoch})
    plt.close()


def msssim_per_image(ms_ssim_fn, pred, target):
    """Per-image MS-SSIM computed in float64.

    On Ampere/Ada GPUs (e.g. L40S) PyTorch runs convolutions in TF32 by default.
    pytorch_msssim computes variances as gauss(X*X) - mu^2, and the TF32 error there
    is about the size of C2 = 9e-4, so on smooth images the contrast term (and the
    final score) can come out > 1. float64 convs never use TF32. Gradients still flow."""
    return ms_ssim_fn(pred.double(), target.double()).float()


def psnr_per_image(pred, target):
    mse = ((pred - target) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def evaluate_bicubic_baseline(loader, ms_ssim_fn, scale_factor, device):
    """MS-SSIM / PSNR of a plain bicubic upscale -- the number the model has to beat."""
    total_ssim, total_psnr, n = 0.0, 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        total_ssim += msssim_per_image(ms_ssim_fn, up, hr).sum().item()
        total_psnr += psnr_per_image(up, hr).sum().item()
        n += lr.size(0)
    return total_ssim / n, total_psnr / n


def save_checkpoint(path, model, optimizer, epoch, best_metric, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        },
        path
    )
    print(f"Checkpoint saved at {path} with MS-SSIM: {best_metric:.6f}", flush=True)


def main(dataset, data_root="dataset", scale_factor=4, preprocess="crop", downscale_lr=False,
         train_crops_per_image=8, val_crops_per_image=5):
    run = wandb.init(
        entity="serena-agarwal-brown-university",
        project="srn-model",
        name=f"{dataset}-final",
        group=dataset,
        config={
            "dataset": dataset,
            "learning_rate": 0.00019480875188999,
            "epochs": 30,
            "batch_size": 4,
            "hr_crop_size": 256,
            "scale_factor": scale_factor,
            "preprocess": preprocess,
            "downscale_lr": downscale_lr,
            "train_crops_per_image": train_crops_per_image,
            "val_crops_per_image": val_crops_per_image,
            "alpha_content": 0.5833495368611317,
            "lambda_resnet": 0.00603659673357171,
            "perceptual_layer": "block7",
            "min_lr_ratio": 0.01,  # cosine schedule decays lr to learning_rate * this
        }
    )
    config = run.config

    # Plot every metric against our own "epoch" value so train and
    # validation points line up on the same x-axis.
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # Validation uses is_train=False: fixed center crop, no flips/rotations/noise.
    # Layout: <data_root>/train/<dataset>_lr, <data_root>/train/<dataset>_hr (same for val)
    if preprocess == "resize":
        # Old behavior: LR resized to 128x128, HR used as-is (for kaggle)
        train_dataset = ResizePairDataset(
            lr_dir=f"{data_root}/train/{dataset}_lr",
            hr_dir=f"{data_root}/train/{dataset}_hr",
        )
        test_dataset = ResizePairDataset(
            lr_dir=f"{data_root}/val/{dataset}_lr",
            hr_dir=f"{data_root}/val/{dataset}_hr",
        )
    else:
        train_dataset = ImagePairDataset(
            lr_dir=f"{data_root}/train/{dataset}_lr",
            hr_dir=f"{data_root}/train/{dataset}_hr",
            hr_crop_size=config.hr_crop_size,
            scale_factor=config.scale_factor,
            is_train=True,
            downscale_lr=config.downscale_lr,
            crops_per_image=config.train_crops_per_image,
        )
        test_dataset = ImagePairDataset(
            lr_dir=f"{data_root}/val/{dataset}_lr",
            hr_dir=f"{data_root}/val/{dataset}_hr",
            hr_crop_size=config.hr_crop_size,
            scale_factor=config.scale_factor,
            is_train=False,
            downscale_lr=config.downscale_lr,
            crops_per_image=config.val_crops_per_image,
        )

    # The model upsamples by a fixed factor (PixelShuffle), so take it from the data.
    sample_lr, sample_hr = test_dataset[0]
    model_scale = sample_hr.shape[-1] // sample_lr.shape[-1]
    assert sample_lr.shape[-1] * model_scale == sample_hr.shape[-1] and \
        sample_lr.shape[-2] * model_scale == sample_hr.shape[-2], (
        f"HR {tuple(sample_hr.shape)} is not an integer multiple of LR {tuple(sample_lr.shape)}"
    )
    print(f"Dataset created! train={len(train_dataset)} val={len(test_dataset)} crops, "
          f"LR {tuple(sample_lr.shape)} -> HR {tuple(sample_hr.shape)} (x{model_scale})", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("Data loaded!", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRN(num_channels=1, nf=64, scale_factor=model_scale).to(device)

    # Load in perceptual loss model
    num_classes = 23
    resnet = ResNet(3, num_classes)
    checkpoint = torch.load("saved_models/new_resnet.pth", map_location=device)
    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet = resnet.to(device)
    resnet.eval()
    feature_extractor = FeatureExtractor(resnet, up_to=config.perceptual_layer).to(device)
    perceptual_loss = PerceptualLoss(feature_extractor).to(device)

    print("Model and device initialized!", flush=True)
    print(f"Device is: {device}!", flush=True)

    l1_criterion = nn.L1Loss()
    # size_average=False -> one score per image; always call through msssim_per_image (float64)
    ms_ssim_fn = MS_SSIM(data_range=1.0, size_average=False, channel=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.learning_rate * config.min_lr_ratio
    )

    print("Initialized loss and optimizer!", flush=True)

    bicubic_ms_ssim, bicubic_psnr = evaluate_bicubic_baseline(test_loader, ms_ssim_fn, model_scale, device)
    run.summary["bicubic msssim"] = bicubic_ms_ssim
    run.summary["bicubic psnr"] = bicubic_psnr
    print(f"Bicubic baseline | MS-SSIM: {bicubic_ms_ssim:.6f} | PSNR: {bicubic_psnr:.3f} dB", flush=True)

    best_ms_ssim = 0.0
    num_images_to_save = 5

    for i in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_resnet_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0

        for images, masks in train_loader:

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            ms_ssim_loss = 1.0 - msssim_per_image(ms_ssim_fn, outputs, masks).mean()
            l1_loss = l1_criterion(outputs, masks)

            content_loss = config.alpha_content * ms_ssim_loss + (1.0 - config.alpha_content) * l1_loss

            outputs_for_res = prep_for_resnet(outputs)
            masks_for_res = prep_for_resnet(masks)
            resnet_loss = perceptual_loss(outputs_for_res, masks_for_res)

            total = content_loss + config.lambda_resnet * resnet_loss

            optimizer.zero_grad()
            total.backward()

            # clip_grad_norm_ returns the norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += total.item() * images.size(0)
            total_resnet_loss += resnet_loss.item() * images.size(0)
            total_grad_norm += grad_norm.item()
            num_batches += 1

        avg_loss = total_loss / len(train_dataset)
        avg_resnet_loss = total_resnet_loss / len(train_dataset)
        avg_grad_norm = total_grad_norm / num_batches

        wandb.log({
            "epoch": i + 1,
            "loss": avg_loss,
            "resnet loss": avg_resnet_loss,
            "total norm": avg_grad_norm,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"Epoch {i + 1}/{config.epochs} | Average Training Loss: {avg_loss:.6f}", flush=True)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        ms_ssim_score = 0.0
        psnr_score = 0.0
        max_ms_ssim = 0.0
        images_saved = 0

        with torch.no_grad():
            for batch_idx, (val_images, val_masks) in enumerate(test_loader):
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)

                if images_saved < num_images_to_save:
                    num_to_save = min(num_images_to_save - images_saved, val_images.size(0))
                    for img_idx in range(num_to_save):
                        save_image_comparison(
                            input_img=val_images[img_idx],
                            output_img=val_outputs[img_idx],
                            target_img=val_masks[img_idx],
                            epoch=i + 1,
                            idx=images_saved,
                            dataset=dataset,
                        )
                        images_saved += 1

                scores = msssim_per_image(ms_ssim_fn, val_outputs, val_masks)
                ms_ssim_score += scores.sum().item()
                val_loss += (1.0 - scores).sum().item()
                max_ms_ssim = max(max_ms_ssim, scores.max().item())
                psnr_score += psnr_per_image(val_outputs, val_masks).sum().item()

        avg_ms_ssim = ms_ssim_score / len(test_dataset)
        avg_val_loss = val_loss / len(test_dataset)
        avg_psnr = psnr_score / len(test_dataset)

        wandb.log({
            "epoch": i + 1,
            "msssim-score": avg_ms_ssim,
            "val loss": avg_val_loss,
            "val psnr": avg_psnr,
            "max per-image msssim": max_ms_ssim,
            "msssim gain over bicubic": avg_ms_ssim - bicubic_ms_ssim,
            "psnr gain over bicubic": avg_psnr - bicubic_psnr,
        })
        print(f"Epoch {i + 1}/{config.epochs} | Validation Loss: {avg_val_loss:.4f} | MS-SSIM score: {avg_ms_ssim:.6f} "
              f"(bicubic {bicubic_ms_ssim:.6f}) | PSNR: {avg_psnr:.3f} dB (bicubic {bicubic_psnr:.3f})", flush=True)
        if max_ms_ssim > 1.0 + 1e-6:
            print(f"WARNING: a validation image scored MS-SSIM {max_ms_ssim:.6f} > 1 -- metric is still numerically off", flush=True)

        if avg_ms_ssim > best_ms_ssim:
            best_ms_ssim = avg_ms_ssim
            checkpoint_path = f"saved_models/srn_{dataset}_best_epoch{i+1}_msssim{avg_ms_ssim:.6f}.pth"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=i + 1,
                best_metric=best_ms_ssim,
                config=run.config
            )
            print(f"New best MS-SSIM: {best_ms_ssim:.6f} - Model saved!", flush=True)

    print(f"\nTraining complete! Best MS-SSIM achieved: {best_ms_ssim:.6f}", flush=True)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset prefix, e.g. dataseta -> expects dataseta_lr and dataseta_hr inside <data_root>/train and <data_root>/val")
    parser.add_argument("--data_root", default="dataset",
                        help="Folder containing train/ and val/ (default: dataset)")
    parser.add_argument("--scale_factor", type=int, default=4,
                        help="HR size / LR size for this dataset (default: 4)")
    parser.add_argument("--preprocess", choices=["crop", "resize"], default="crop",
                        help="crop = random/center crops (default); resize = old behavior (LR resized to 128x128, HR as-is)")
    parser.add_argument("--downscale_lr", action="store_true",
                        help="LR is stored at the same size as HR but is really scale_factor x lower resolution (e.g. kaggle): shrink it first")
    parser.add_argument("--train_crops_per_image", type=int, default=8,
                        help="random crops taken from each training image per epoch (default: 8)")
    parser.add_argument("--val_crops_per_image", type=int, default=5,
                        help="fixed validation crops per image, 1-5 (default: 5)")
    args = parser.parse_args()
    main(args.dataset, args.data_root, args.scale_factor, args.preprocess, args.downscale_lr,
         args.train_crops_per_image, args.val_crops_per_image)