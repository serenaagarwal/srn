import wandb
from run_model import main

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

sweep_id = wandb.sweep(
    sweep_config,
    entity="serena-agarwal-brown-university",
    project="srn-model",
)
wandb.agent(sweep_id, function=main, count=20)  # try 20 combinations