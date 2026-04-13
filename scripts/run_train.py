import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.config import Config
from meshgraphnet.train_eval import train
from meshgraphnet.normalization import get_stats


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_cfg_dict(cfg):
    return {
        "device": cfg.device,
        "model": {
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
        },
        "training": {
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "num_epochs": cfg.training.num_epochs,
        },
        "data": {
            "noise_scale": cfg.data.noise_scale,
            "noise_gamma": cfg.data.noise_gamma,
        },
    }

def main():
    cfg = Config()

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model.hidden_dim = 10
    cfg.model.num_layers = 10
    cfg.training.batch_size = 16
    cfg.training.learning_rate = 1e-3
    cfg.training.weight_decay = 5e-4
    cfg.training.num_epochs = 1000

    # optional repo-style data noise settings
    if not hasattr(cfg, "data"):
        class DataConfig:
            noise_scale = 0.0
            noise_gamma = 1.0
        cfg.data = DataConfig()
    else:
        if not hasattr(cfg.data, "noise_scale"):
            cfg.data.noise_scale = 0.0
        if not hasattr(cfg.data, "noise_gamma"):
            cfg.data.noise_gamma = 1.0

    set_seed(5)

    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # file_path = os.path.join(base_dir, "meshgraphnets_miniset5traj_vis.pt")
    # data_all = torch.load(file_path, weights_only=False)
    # train_size = 45
    # valid_size = 10

    # # repo-style: shuffle first, then split
    # random.shuffle(data_all)
    # data_train = data_all[:train_size]
    # data_valid = data_all[train_size: train_size + valid_size]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(base_dir, "data", "processed", "data_pt", "train.pt")
    valid_path = os.path.join(base_dir, "data", "processed", "data_pt", "valid.pt")

    data_train = torch.load(train_path, weights_only=False)
    data_train = data_train[0:600]
    data_valid = torch.load(valid_path, weights_only=False)
    data_valid = data_valid[0:50]

    # optional: shuffle training samples only
    random.shuffle(data_train)

    print(f"Train samples: {len(data_train)}")
    print(f"Valid samples: {len(data_valid)}")
    print(f"Train sample x shape: {data_train[0].x.shape}")
    print(f"Train sample edge_index shape: {data_train[0].edge_index.shape}")
    print(f"Train sample edge_attr shape: {data_train[0].edge_attr.shape}")
    print(f"Train sample y shape: {data_train[0].y.shape}")
    print(f"Valid sample x shape: {data_valid[0].x.shape}")
    print(f"Device: {cfg.device}")

    # compute stats on train, originally we had train+valid but that leaks valid info into train stats
    stats_list = get_stats(data_train)

    (
        model,
        train_losses,
        valid_losses,
        velocity_valid_losses,
        best_model,
        best_valid_loss,
    ) = train(
        data_train=data_train,
        data_valid=data_valid,
        stats_list=stats_list,
        cfg=cfg,
    )

    print("Finished training.")
    print(f"Best valid loss: {best_valid_loss:.6f}")

    plot_dir = os.path.join("outputs", "figures")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(
        os.path.join(plot_dir, "loss_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    save_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    cfg_dict = build_cfg_dict(cfg)

    if isinstance(best_model, dict):
        best_state_dict = best_model
    else:
        best_state_dict = best_model.state_dict()

    checkpoint = {
        "model_state_dict": best_state_dict,
        "stats_list": stats_list,
        "cfg_dict": cfg_dict,
        "num_node_features": data_train[0].x.shape[1],
        "num_edge_features": data_train[0].edge_attr.shape[1],
        "num_classes": 2,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "velocity_valid_losses": velocity_valid_losses,
        "best_valid_loss": best_valid_loss,
    }

    save_path = os.path.join(save_dir, "meshgraphnet_train600_valid50.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved best model checkpoint to {save_path}")


if __name__ == "__main__":
    main()