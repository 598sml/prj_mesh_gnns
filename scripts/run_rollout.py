import os
import sys
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.inference import (
    load_checkpoint_and_model,
    rollout_one_trajectory,
)
from meshgraphnet.data_utils import (
    build_ordered_test_data,
)

from meshgraphnet.plot_utils import (
    make_comparison_plot,
    save_rmse_plot,
)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(5)
    delta_t = 0.01

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # checkpoint_path = os.path.join(
    #     base_dir, "outputs", "checkpoints", "meshgraphnet_first_run.pt"
    # )
    checkpoint_path = os.path.join(
        base_dir, "outputs", "checkpoints", "meshgraphnet_train600_valid50.pt"
    )

    # file_path = os.path.join(base_dir, "meshgraphnets_miniset5traj_vis.pt")
    # test_data_path = os.path.join(base_dir, "test_ordered_debug.pt")

    # # Ordered test slice from the original dataset
    # test_start = 45
    # test_size = 10
    # test_data = build_ordered_test_data(
    #     file_path=file_path,
    #     test_start=test_start,
    #     test_size=test_size,
    #     save_path=test_data_path,
    # )

    # test_data_path = os.path.join(
    #     base_dir, "data", "processed", "data_pt", "test.pt"
    # )

    # print("checkpoint_path:", checkpoint_path)
    # print("test_data_path:", test_data_path)
    # print("checkpoint exists:", os.path.exists(checkpoint_path))
    # print("test exists:", os.path.exists(test_data_path))

    # checkpoint, cfg, model, stats = load_checkpoint_and_model(checkpoint_path)
    # print(f"Using device: {cfg.device}")

    # test_data = torch.load(test_data_path, weights_only=False)
    # print(f"Loaded {len(test_data)} ordered test graph samples.")

    split = "train"   # "train" or "test"
    traj_idx = 0      # only used when split == "train"

    if split == "test":
        rollout_data_path = os.path.join(
            base_dir, "data", "processed", "data_pt", "test.pt"
        )
        rollout_data = torch.load(rollout_data_path, weights_only=False)

        print("checkpoint_path:", checkpoint_path)
        print("rollout_data_path:", rollout_data_path)
        print("checkpoint exists:", os.path.exists(checkpoint_path))
        print("rollout data exists:", os.path.exists(rollout_data_path))
        print(f"Loaded {len(rollout_data)} ordered test graph samples.")

    elif split == "train":
        train_data_path = os.path.join(
            base_dir, "data", "processed", "data_pt", "train.pt"
        )
        train_all = torch.load(train_data_path, weights_only=False)

        start = traj_idx * 599
        end = (traj_idx + 1) * 599
        rollout_data = train_all[start:end]

        print("checkpoint_path:", checkpoint_path)
        print("train_data_path:", train_data_path)
        print("checkpoint exists:", os.path.exists(checkpoint_path))
        print("train data exists:", os.path.exists(train_data_path))
        print(f"Loaded train trajectory {traj_idx} with {len(rollout_data)} graph samples.")

    else:
        raise ValueError(f"Unknown split: {split}")

    checkpoint, cfg, model, stats = load_checkpoint_and_model(checkpoint_path)
    print(f"Using device: {cfg.device}")

    pred_graphs, rollout_rmse = rollout_one_trajectory(
        model=model,
        test_data=rollout_data,
        stats=stats,
        delta_t=delta_t,
        device=cfg.device,
    )

    print("\nRollout RMSE by step:")
    for i, rmse in enumerate(rollout_rmse):
        print(f"rollout step {i} -> {i+1}: {rmse:.6f}")

    out_dir = os.path.join(base_dir, "outputs", "2rollout_figures")
    os.makedirs(out_dir, exist_ok=True)

    rmse_plot_path = os.path.join(out_dir, "rollout_rmse.png")
    save_rmse_plot(
        rmse_values=rollout_rmse,
        save_path=rmse_plot_path,
        xlabel="Rollout step index",
        ylabel="Velocity RMSE",
        title="Free-running rollout RMSE",
    )
    print(f"Saved rollout RMSE plot to: {rmse_plot_path}")

    # sample_steps = [1, 2, 5, 8]

    sample_steps = [1, 2, 3, 4, 5, 10, 50, 100, 200, 300, 400, 500]
    for step in sample_steps:
        if step >= len(rollout_data):
            continue

        pred_graph = pred_graphs[step]
        true_graph = rollout_data[step]

        pred_u = pred_graph.x[:, 0].cpu()
        pred_v = pred_graph.x[:, 1].cpu()
        true_u = true_graph.x[:, 0].cpu()
        true_v = true_graph.x[:, 1].cpu()

        err_u = pred_u - true_u
        err_v = pred_v - true_v

        make_comparison_plot(
            mesh_pos=true_graph.mesh_pos.cpu(),
            cells=true_graph.cells.cpu(),
            true_field=true_u,
            pred_field=pred_u,
            error_field=err_u,
            component_name="u",
            title_prefix_pred="Rollout pred",
            title_suffix=f"(step {step})",
            save_path=os.path.join(out_dir, f"rollout_step_{step:03d}_u.png"),
        )

        make_comparison_plot(
            mesh_pos=true_graph.mesh_pos.cpu(),
            cells=true_graph.cells.cpu(),
            true_field=true_v,
            pred_field=pred_v,
            error_field=err_v,
            component_name="v",
            title_prefix_pred="Rollout pred",
            title_suffix=f"(step {step})",
            save_path=os.path.join(out_dir, f"rollout_step_{step:03d}_v.png"),
        )


        print(f"Saved rollout comparison plots for step {step}")

    print(f"\nSaved rollout figures to: {out_dir}")


if __name__ == "__main__":
    main()