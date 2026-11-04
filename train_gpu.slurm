#!/bin/bash
#SBATCH --job-name=srn_train
#SBATCH --output=logs/srn_%j.out
#SBATCH --error=logs/srn_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /oscar/home/sagarw56/srn/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#python run_model.py
python run_model.py --dataset flickr
python run_model.py --dataset kaggle --downscale_lr
python run_model.py --dataset new_div2k