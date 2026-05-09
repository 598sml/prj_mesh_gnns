import os
import sys
import argparse
import random
import torch
import matplotlib.pyplot as plt
import json
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.utils import load_json_config, set_seed
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

def main():
    # Accept an optional --config argument so that submit_training.sh can pass a
    # frozen copy of config.json created at sbatch submission time. This prevents
    # the job from silently picking up edits made to config.json while it was
    # sitting in the SLURM queue. Without this, every queued job reads the live
    # file at the moment the node actually starts — not at the moment you submitted.
    # Falls back to configs/config.json when running locally without sbatch.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "config.json"),
        help="Path to the config JSON file. Pass a frozen copy when submitting via sbatch.",
    )
    args = parser.parse_args()
    config_path = args.config

    cfg_json = load_json_config(config_path)
    data_cfg = cfg_json[cfg_json["dataset_source"]]

    device = "cuda" if (cfg_json["device"] == "cuda" and torch.cuda.is_available()) else "cpu"

    if cfg_json["wandb"]["enabled"]:
        wandb.init(
            project=cfg_json["wandb"]["project"],
            name=cfg_json["wandb"]["run_name"],
            config=cfg_json,
        )

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
    print(f"Device: {device}")

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
        cfg_json=cfg_json,
        device=device,
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

    cfg_dict = {
        "device": device,
        "model": cfg_json["model"],
        "training": cfg_json["training"],
        "data": cfg_json["data"],
    }

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