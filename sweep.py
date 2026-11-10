import argparse
import wandb
from run_model import main

# wandb.agent calls its function with no arguments, so the command-line options
# have to be parsed here and passed into main() for every sweep run.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="dataset prefix, e.g. flickr2")
parser.add_argument("--data_root", default="dataset")
parser.add_argument("--scale_factor", type=int, default=4)
parser.add_argument("--preprocess", choices=["crop", "resize"], default="crop")
parser.add_argument("--downscale_lr", action="store_true")
parser.add_argument("--train_crops_per_image", type=int, default=8)
parser.add_argument("--val_crops_per_image", type=int, default=5)
parser.add_argument("--count", type=int, default=4,
                    help="sweep runs in THIS job (default 4: ~10 epochs x ~13 min each fits in 12 h)")
parser.add_argument("--sweep_id", default=None,
                    help="join an existing sweep instead of creating a new one (printed by the first job)")
args = parser.parse_args()


def train():
    main(args.dataset, args.data_root, args.scale_factor, args.preprocess, args.downscale_lr,
         args.train_crops_per_image, args.val_crops_per_image)


sweep_config = {
    "method": "bayes",
    "metric": {"name": "msssim-score", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size":    {"value": 4},
        "alpha_content": {"min": 0.5, "max": 0.95},
        "lambda_resnet": {"distribution": "log_uniform_values", "min": 1e-3, "max": 1e-1},
        "epochs":        {"value": 10},   # shorter runs just for searching
        "hr_crop_size":  {"value": 256},  # keep fixed
    },
    # kills runs that are clearly losing, which saves compute
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

if args.sweep_id:
    sweep_id = args.sweep_id
else:
    sweep_id = wandb.sweep(
        sweep_config,
        entity="serena-agarwal-brown-university",
        project="srn-model",
    )
print(f"Sweep ID: {sweep_id}  (pass --sweep_id {sweep_id} to add more runs to this sweep)", flush=True)
wandb.agent(sweep_id, function=train, count=args.count,
            entity="serena-agarwal-brown-university", project="srn-model")