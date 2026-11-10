#!/bin/bash
#SBATCH --job-name=make_sr_pairs
#SBATCH --output=logs/pairs_%j.out
#SBATCH --error=logs/pairs_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

SRC=/oscar/home/sagarw56/srn/flickr2k

source /oscar/home/sagarw56/srn/venv/bin/activate
cd /oscar/home/sagarw56/srn

python make_sr_pairs.py --src "$SRC" --name flickr2 --hr_downscale 4 --scale 4 --val_start 2000
