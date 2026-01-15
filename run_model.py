import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from preprocess import ImagePairDataset
from rrdb_arch import SRN
from pytorch_msssim import MS_SSIM
from resnet import ResNet
from perceptual_loss import FeatureExtractor, PerceptualLoss, prep_for_resnet
import matplotlib.pyplot as plt
import wandb


run = wandb.init(entity="serena-agarwal-brown-university", project="srn-model", config={
    "learning_rate": .001, 
    "epochs": 10}
)

def save_image_comparison(input_img, output_img, target_img, epoch, idx, save_dir="visual_outputs"):
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
    plt.savefig(f"{save_dir}/epoch{epoch}_sample{idx}.png")

    run.log({f"epoch{epoch}_sample{idx}": wandb.Image(f"{save_dir}/epoch{epoch}_sample{idx}.png")})

    plt.close()

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


def main():
    
    train_dataset = ImagePairDataset("dataset/train")
    test_dataset = ImagePairDataset("dataset/val")

    #train_subset = Subset(train_dataset, list(range(50)))

    print("dataset created!", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("data loaded!", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRN(1, 64, 256).to(device)

    #load in perceptual loss model
    num_classes = 23
    resnet = ResNet(3, num_classes)
    checkpoint = torch.load("saved_models/new_resnet.pth", map_location=device)
    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet = resnet.to(device)
    resnet.eval()
    feature_extractor = FeatureExtractor(resnet).to(device)
    perceptual_loss = PerceptualLoss(feature_extractor).to(device)

    print("model and device initialized!", flush=True)
    print(f"device is: {device}!", flush=True)

    criterion = nn.MSELoss()
    ms_ssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    print("initialized loss and optimizer!", flush=True)

    #training loooop
    best_ms_ssim = 0.0
    num_images_to_save = 5


    epochs = 30

    for i in range(epochs):
        model.train()
        total_loss = 0.0
        total_resnet_loss = 0.0
        lamda_resnet = 0.01

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            mse_loss = criterion(outputs, masks)

            outputs_for_res = prep_for_resnet(outputs)
            masks_for_res = prep_for_resnet(masks)
            resnet_loss = perceptual_loss(outputs_for_res, masks_for_res)
            
            total = mse_loss + lamda_resnet * resnet_loss
            
            optimizer.zero_grad()
            total.backward()

            optimizer.step()

            total_loss += total.item() * images.size(0)
            total_resnet_loss += resnet_loss.item() * images.size(0)
            print("done!", flush=True)

            # if i == 9:
            #     image = Image.fromarray(images)
            #     image.save(format='PNG')
            
        avg_loss = total_loss / len(train_dataset)
        avg_resnet_loss = total_resnet_loss / len(train_dataset)

        total_norm = 0.0
        squared_param_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                squared_param_norm = p.grad.data.norm(2)
                total_norm += squared_param_norm.item() ** 2
            total_norm = total_norm ** (0.5)

        run.log({"loss": avg_loss, "total norm": total_norm, "squared param norm": squared_param_norm, "resnet loss": avg_resnet_loss})
        
        print(f"Average loss for epoch {i + 1} was {avg_loss}.", flush=True)

        #validation!

        model.eval()
        val_loss = 0.0
        ms_ssim_score = 0.0
        val_len = len(test_dataset) / epochs
        images_saved = 0

        with torch.no_grad():
            for batch_idx, (val_images, val_masks) in enumerate(test_loader):
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                val_outputs = model(val_images)

                if images_saved < num_images_to_save:
                    num_to_save = min(num_images_to_save - images_saved, val_images.shape[0])
                    for img_idx in range(num_to_save):
                        save_image_comparison(
                            input_img=val_images[img_idx],
                            output_img=val_outputs[img_idx],
                            target_img=val_masks[img_idx],
                            epoch=i,
                            idx=images_saved
                        )
                        images_saved += 1
                loss = criterion(val_outputs, val_masks)

                score = ms_ssim_fn(val_outputs, val_masks)
            
                ms_ssim_score += score.item() * val_images.size(0)
                val_loss += loss.item() * val_images.size(0)
        avg_ms_ssim = ms_ssim_score / len(test_dataset)
        avg_val_loss = val_loss / len(test_dataset) 

        run.log({"msssim-score": avg_ms_ssim, "val loss": avg_val_loss, "training_loss": avg_loss})
        print(f"Epoch {i + 1}/{epochs} | Validation Loss: {avg_val_loss:.4f} | MS-SSIM score: {avg_ms_ssim:.6f}", flush=True)
        
        if avg_ms_ssim > best_ms_ssim:
            best_ms_ssim = avg_ms_ssim
            checkpoint_path = f"saved_models/srn_best_epoch{i+1}_msssim{avg_ms_ssim:.6f}.pth"
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
    main()