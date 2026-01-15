#!/bin/bash
#SBATCH --job-name=srn_train
#SBATCH --output=logs/srn_%j.out
#SBATCH --error=logs/srn_%j.err
#SBATCH --account=carney-dborton-condo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /oscar/home/sagarw56/srn/venv/bin/activate

python run_model.py