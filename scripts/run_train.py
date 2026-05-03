import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.config import Config
from meshgraphnet.train_eval import train
from meshgraphnet.normalization import get_stats

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def load_json_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def apply_json_to_cfg(cfg, cfg_json):
    cfg.device = (
        "cuda"
        if (cfg_json["device"] == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    cfg.model.hidden_dim = cfg_json["model"]["hidden_dim"]
    cfg.model.num_layers = cfg_json["model"]["num_layers"]

    cfg.training.batch_size = cfg_json["training"]["batch_size"]
    cfg.training.learning_rate = cfg_json["training"]["learning_rate"]
    cfg.training.weight_decay = cfg_json["training"]["weight_decay"]
    cfg.training.num_epochs = cfg_json["training"]["num_epochs"]

    if not hasattr(cfg, "data"):
        class DataConfig:
            pass
        cfg.data = DataConfig()

    cfg.data.noise_scale = cfg_json["data"]["noise_scale"]
    cfg.data.noise_gamma = cfg_json["data"]["noise_gamma"]

    return cfg

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

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.json")
    cfg_json = load_json_config(config_path)
    data_cfg = cfg_json[cfg_json["dataset_source"]]

    if cfg_json["wandb"]["enabled"]:
        wandb.init(
            project=cfg_json["wandb"]["project"],
            name=cfg_json["wandb"]["run_name"],
            config=cfg_json,
        )
    cfg = apply_json_to_cfg(cfg, cfg_json)

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

    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # file_path = os.path.join(base_dir, "meshgraphnets_miniset5traj_vis.pt")
    # data_all = torch.load(file_path, weights_only=False)
    # train_size = 45
    # valid_size = 10

    # # repo-style: shuffle first, then split
    # random.shuffle(data_all)
    # data_train = data_all[:train_size]
    # data_valid = data_all[train_size: train_size + valid_size]

    set_seed(cfg_json["seed"])

    base_dir = PROJECT_ROOT

    log_dir = os.path.join(
        base_dir,
        cfg_json["paths"]["checkpoint_dir"],
        cfg_json["experiment_name"],
    )
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "training.log")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    if cfg_json["dataset_source"] == "dataset_colab":
        file_path = os.path.join(base_dir, cfg_json["dataset_colab"]["file_path"])
        data_all = torch.load(file_path, weights_only=False)

        # colab-style: shuffle first, then split
        if cfg_json["dataset_colab"]["shuffle_before_split"]:
            random.shuffle(data_all)

        train_size = cfg_json["dataset_colab"]["train_size"]
        valid_size = cfg_json["dataset_colab"]["valid_size"]

        data_train = data_all[:train_size]
        data_valid = data_all[train_size: train_size + valid_size]

    else:
        train_path = os.path.join(base_dir, data_cfg["train_data"])
        valid_path = os.path.join(base_dir, data_cfg["valid_data"])

        data_train = torch.load(train_path, weights_only=False)
        data_valid = torch.load(valid_path, weights_only=False)

        if data_cfg["use_subset"]:
            data_train = data_train[: data_cfg["train_subset_size"]]
            data_valid = data_valid[: data_cfg["valid_subset_size"]]

        # optional: shuffle training samples only
        if data_cfg["shuffle_train"]:
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
        cfg_json=cfg_json,
    )

    print("Finished training.")
    print(f"Best valid loss: {best_valid_loss:.6f}")

    plot_dir = os.path.join(
        base_dir,
        cfg_json["paths"]["figure_dir"],
        cfg_json["experiment_name"],
    )
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

    save_dir = os.path.join(
        base_dir,
        cfg_json["paths"]["checkpoint_dir"],
        cfg_json["experiment_name"],
    )
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

    save_path = os.path.join(save_dir, f"{cfg_json['experiment_name']}.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved best model checkpoint to {save_path}")

    config_save_path = os.path.join(save_dir, "config_used.json")
    with open(config_save_path, "w") as f:
        json.dump(cfg_json, f, indent=2)

    print(f"Saved config to {config_save_path}")

    sys.stdout = original_stdout
    log_file.close()
    print(f"Saved training log to {log_path}")
    if cfg_json["wandb"]["enabled"]:
        wandb.finish()


if __name__ == "__main__":
    main()