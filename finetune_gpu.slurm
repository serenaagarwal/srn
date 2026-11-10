#!/bin/bash
#SBATCH --job-name=srn_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Usage: sbatch finetune_gpu.slurm <lambda_seg>
#   sbatch finetune_gpu.slurm 0      -> no segmentation loss (control)
#   sbatch finetune_gpu.slurm 0.01   -> with segmentation loss
LAMBDA_SEG=${1:?pass lambda_seg as the first argument, e.g. sbatch finetune_gpu.slurm 0.01}

source /oscar/home/sagarw56/srn/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /oscar/home/sagarw56/srn

python finetune.py \
    --checkpoint saved_models/srn_flickr2_best_epoch14_msssim0.943895.pth \
    --lambda_seg "$LAMBDA_SEG" \
    --lambda_resnet 0
