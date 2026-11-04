import os
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from edgemap_finetune_preprocess import ImagePairDataset
from edgemap_rrdb_arch import SRN
from pytorch_msssim import MS_SSIM
from resnet import ResNet
from seg_loss_unet import U_Net
from perceptual_loss import FeatureExtractor, PerceptualLoss, prep_for_resnet
import matplotlib.pyplot as plt
import wandb
import sys


# FINE-TUNING SETTINGS

FINETUNE_CHECKPOINT = "model1info/best_model_paths/srn_best_epoch23_msssim0.993069.pth"
FINETUNE_LEARNING_RATE = 1e-4
FINETUNE_EPOCHS = 20 

run = wandb.init(entity="serena-agarwal-brown-university", project="srn-model", config={
    "learning_rate": FINETUNE_LEARNING_RATE,
    "epochs": FINETUNE_EPOCHS,
    "finetune_from": FINETUNE_CHECKPOINT,
})

def save_image_comparison(input_img, output_img, target_img, epoch, idx, save_dir="edge_map_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    lr_img = input_img[0].squeeze().detach().cpu().numpy()
    edge_map = input_img[1].squeeze().detach().cpu().numpy()
    output_img = output_img.squeeze().detach().cpu().numpy()
    target_img = target_img.squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(lr_img, cmap='gray')
    axs[0].set_title("Low-Res Input")
    axs[0].axis('off')

    axs[1].imshow(edge_map, cmap='gray')
    axs[1].set_title("Low-Res Input")
    axs[1].axis('off')

    axs[2].imshow(output_img, cmap='gray')
    axs[2].set_title("SRN Output")
    axs[2].axis('off')

    axs[3].imshow(target_img, cmap='gray')
    axs[3].set_title("High-Res Ground Truth")
    axs[3].axis('off')

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

def load_and_adapt_checkpoint(model, checkpoint_path, device):
    """
    Safely loads a 1-channel checkpoint into a 2-channel SRN model by copying
    pre-trained weights into Channel 0 (LR image) and initializing Channel 1 (Edge Map).
    Also adjusts optimizer state buffers if present.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    first_conv_weight = state_dict["first_conv.weight"]

    # Check if checkpoint weight was trained on 1 input channel
    if first_conv_weight.shape[1] == 1:
        print("Adapting 1-channel pre-trained weights to 2-channel model input...", flush=True)
        # Create a new weight tensor of shape [64, 2, 3, 3]
        new_first_conv_weight = torch.zeros((64, 2, 3, 3), device=device)
        
        # Copy original 1-channel weights into Channel 0
        new_first_conv_weight[:, 0:1, :, :] = first_conv_weight
        
        # Kaiming normal initialization for Channel 1 (Edge map)
        nn.init.kaiming_normal_(new_first_conv_weight[:, 1:2, :, :], mode='fan_out', nonlinearity='leaky_relu')
        
        # Overwrite weight in state dict
        state_dict["first_conv.weight"] = new_first_conv_weight

        # Clear optimizer state_dict if it contains outdated 1-channel momentum shapes
        if "optimizer_state_dict" in checkpoint:
            print("Resetting optimizer state to match adapted layer shapes...", flush=True)
            checkpoint["optimizer_state_dict"] = None

    model.load_state_dict(state_dict)
    return checkpoint

def main():
    
    train_dataset = ImagePairDataset("borton_data/train")
    test_dataset = ImagePairDataset("borton_data/test")

    #train_subset = Subset(train_dataset, list(range(50)))

    print("dataset created!", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SRN(2, 64, 512).to(device)

    #load in perceptual loss model
    num_classes = 23
    resnet = ResNet(3, num_classes)
    resnet_checkpoint = torch.load("saved_models/new_resnet.pth", map_location=device)
    resnet.load_state_dict(resnet_checkpoint["model_state_dict"])
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    resnet_feature_extractor = FeatureExtractor(resnet).to(device)
    perceptual_loss = PerceptualLoss(resnet_feature_extractor).to(device)

    #load in segmentation loss model
    u_net = U_Net(64)
    u_net_checkpoint = torch.load("saved_models/new_unet.pth", map_location=device)
    u_net.load_state_dict(u_net_checkpoint["model_state_dict"])
    u_net = u_net.to(device)
    u_net.eval()
    for p in u_net.parameters():
        p.requires_grad = False

    print("model and device initialized!", flush=True)
    print(f"device is: {device}!", flush=True)

    #load in pre-trained checkpoint
    finetune_checkpoint = load_and_adapt_checkpoint(model, FINETUNE_CHECKPOINT, device)
    print(f"Loaded pretrained weights from {FINETUNE_CHECKPOINT}", flush=True)

    start_epoch = finetune_checkpoint.get("epoch", 0)
    best_ms_ssim = finetune_checkpoint.get("best_metric", 0.0)
    print(f"Resuming from epoch {start_epoch}, previous best MS-SSIM: {best_ms_ssim:.6f}", flush=True)

    criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    ms_ssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss()
    
    #learning rate
    # NEW / FIXED CODE
    opt_state = finetune_checkpoint.get("optimizer_state_dict")
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
            for param_group in optimizer.param_groups:
                param_group["lr"] = FINETUNE_LEARNING_RATE
            print("Successfully loaded optimizer state.", flush=True)
        except Exception as e:
            print(f"Could not load optimizer state due to architecture change ({e}). Starting fresh optimizer.", flush=True)
    else:
        print("Starting Adam optimizer with fresh momentum state for fine-tuning.", flush=True)
    print("initialized loss and optimizer!", flush=True)

    #training loooop
    num_images_to_save = 5

    epochs = FINETUNE_EPOCHS

    for i in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        total_resnet_loss = 0.0
        total_seg_loss = 0.0
        lambda_resnet = 0.01
        lambda_segmentation = 0.01
        alpha_content = .84 #be able to cite!!! from sfsrm

        for images, masks, edges in train_loader:
            images, masks, edges = images.to(device), masks.to(device), edges.to(device)
            inputs = torch.cat([images, edges], dim=1)
            outputs = model(inputs)
            
            sr_seg = u_net(outputs)

            with torch.no_grad():
                gt_seg = u_net(masks).detach()

            seg_loss = bce_loss(sr_seg, gt_seg)

            #mse_loss = criterion(outputs, masks)
            ms_ssim_loss = 1 - ms_ssim_fn(outputs, masks)
            #print("ms_ssim_loss = ", ms_ssim_loss, flush=True)

            l1_loss = l1_criterion(outputs, masks)
            #print("l1_loss = ", l1_loss, flush=True)

            
            content_loss = alpha_content * ms_ssim_loss + (1 - alpha_content) * l1_loss

            outputs_for_res = prep_for_resnet(outputs)
            with torch.no_grad():
                masks_for_res = prep_for_resnet(masks).detach()
            
            resnet_loss = perceptual_loss(outputs_for_res, masks_for_res)
            

            total = content_loss + lambda_resnet * resnet_loss + lambda_segmentation * seg_loss
            
            optimizer.zero_grad()
            total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total.item() * images.size(0)
            total_resnet_loss += resnet_loss.item() * images.size(0)
            total_seg_loss += seg_loss.item() * images.size(0)
            print("done!", flush=True)

            # if i == 9:
            #     image = Image.fromarray(images)
            #     image.save(format='PNG')
            
        avg_loss = total_loss / len(train_dataset)
        avg_resnet_loss = total_resnet_loss / len(train_dataset)
        avg_seg_loss = total_seg_loss / len(train_dataset)

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

        #validation loop here!! 
        with torch.no_grad():
            for batch_idx, (val_images, val_masks, val_edges) in enumerate(test_loader):
                val_images, val_masks, val_edges = val_images.to(device), val_masks.to(device), val_edges.to(device)

                val_inputs = torch.cat([val_images, val_edges], dim=1)
                val_outputs = model(val_inputs)

                if images_saved < num_images_to_save:
                    num_to_save = min(num_images_to_save - images_saved, val_images.shape[0])
                    for img_idx in range(num_to_save):
                        save_image_comparison(
                            input_img=val_inputs[img_idx],
                            output_img=val_outputs[img_idx],
                            target_img=val_masks[img_idx],
                            epoch=i,
                            idx=images_saved
                        )
                        images_saved += 1
                loss = ms_ssim_fn(val_outputs, val_masks)

                score = ms_ssim_fn(val_outputs, val_masks)
            
                ms_ssim_score += score.item() * val_images.size(0)
                val_loss += loss.item() * val_images.size(0)
        avg_ms_ssim = ms_ssim_score / len(test_dataset)
        avg_val_loss = val_loss / len(test_dataset) 

        run.log({"msssim-score": avg_ms_ssim, "val loss": avg_val_loss, "training_loss": avg_loss})
        print(f"Epoch {i + 1}/{start_epoch + epochs} | Validation Loss: {avg_val_loss:.4f} | MS-SSIM score: {avg_ms_ssim:.6f}", flush=True)
        
        #if the new ssim score is better than the previous best one, set that to the new ssim score and save that model!
        if avg_ms_ssim > best_ms_ssim:
            best_ms_ssim = avg_ms_ssim
            checkpoint_path = f"saved_models/srn_finetuned_best_epoch{i+1}_msssim{avg_ms_ssim:.6f}.pth"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=i + 1,
                best_metric=best_ms_ssim,
                config=run.config
            )
            print(f"New best MS-SSIM: {best_ms_ssim:.6f} - Model saved!", flush=True)
    
    print(f"\nFine-tuning complete! Best MS-SSIM achieved: {best_ms_ssim:.6f}", flush=True)    
    run.finish()
    
if __name__ == "__main__":
    main()