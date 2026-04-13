import os
import sys
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.train_eval import evaluate
from meshgraphnet.inference import (
    load_checkpoint_and_model,
    predict_next_velocity,
    reconstruct_true_next_velocity,
    one_step_pair_rmse,
)

from meshgraphnet.data_utils import (
    build_ordered_test_data,
    check_consecutive_pair,
)

from meshgraphnet.plot_utils import (
    make_comparison_plot,
    save_rmse_plot
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

    # # Ordered test slice from the original dataset
    # test_start = 45
    # test_size = 10

    # # Save ordered debug test set
    # test_data_path = os.path.join(base_dir, "test_ordered_debug.pt")
    # test_data = build_ordered_test_data(
    #     file_path=file_path,
    #     test_start=test_start,
    #     test_size=test_size,
    #     save_path=test_data_path,
    # )

    # print("checkpoint_path:", checkpoint_path)
    # print("test_data_path:", test_data_path)
    # print("checkpoint exists:", os.path.exists(checkpoint_path))
    # print("test exists:", os.path.exists(test_data_path))

    test_data_path = os.path.join(base_dir, "data", "processed", "data_pt", "test.pt")

    print("checkpoint_path:", checkpoint_path)
    print("test_data_path:", test_data_path)
    print("checkpoint exists:", os.path.exists(checkpoint_path))
    print("test exists:", os.path.exists(test_data_path))

    # ---------- Load checkpoint ----------
    checkpoint, cfg, model, stats = load_checkpoint_and_model(checkpoint_path)
    print(f"Using device: {cfg.device}")

    # ---------- Load test data ----------
    test_data = torch.load(test_data_path, weights_only=False)
    print(f"Loaded {len(test_data)} test graph samples.")

    # ---------- Dataset consistency check ----------
    check_consecutive_pair(test_data, delta_t, i=0)

    # ---------- Statistics already moved to device in inference.py ----------
    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = stats

    # ---------- Quantitative evaluation ----------
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    test_loss, test_velocity_rmse = evaluate(
        loader=test_loader,
        device=cfg.device,
        model=model,
        mean_vec_x=mean_vec_x,
        std_vec_x=std_vec_x,
        mean_vec_edge=mean_vec_edge,
        std_vec_edge=std_vec_edge,
        mean_vec_y=mean_vec_y,
        std_vec_y=std_vec_y,
        delta_t=delta_t,
    )

    print(f"Test one-step loss: {test_loss:.6f}")
    print(f"Test one-step velocity RMSE: {test_velocity_rmse:.6f}")

    # ---------- Output directory ----------
    plot_dir = os.path.join(base_dir, "outputs", "one_step")
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- Per-step one-step RMSE across ordered sequence ----------
    per_step_rmse = []

    for i in range(len(test_data) - 1):
        sample = test_data[i].to(cfg.device)
        next_sample = test_data[i + 1].to(cfg.device)

        rmse, pred_velocity_next, true_velocity_next = one_step_pair_rmse(
            model=model,
            graph_t=sample,
            graph_tp1=next_sample,
            stats=stats,
            delta_t=delta_t,
        )
        per_step_rmse.append(rmse)

        print(f"step {i} -> {i+1} RMSE: {rmse:.6f}")

    rmse_plot_path = os.path.join(plot_dir, "per_step_rmse.png")
    save_rmse_plot(
        rmse_values=per_step_rmse,
        save_path=rmse_plot_path,
        xlabel="Step index",
        ylabel="Velocity RMSE",
        title="One-step RMSE across ordered test trajectory",
    )

    print(f"Saved per-step RMSE plot to: {rmse_plot_path}")

    # ---------- Multi-sample visualization ----------
    # sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sample_indices = [0, 50, 100, 200, 300, 400, 500]
    for sample_idx in sample_indices:
        if sample_idx >= len(test_data):
            print(f"Skipping sample {sample_idx}: out of range.")
            continue

        sample = test_data[sample_idx].to(cfg.device)

        pred_velocity_next, _ = predict_next_velocity(
            model=model,
            graph=sample,
            stats=stats,
            delta_t=delta_t,
        )
        true_velocity_next = reconstruct_true_next_velocity(
            graph=sample,
            delta_t=delta_t,
        )
        error_velocity = pred_velocity_next - true_velocity_next

        make_comparison_plot(
            mesh_pos=sample.mesh_pos.detach().cpu(),
            cells=sample.cells.detach().cpu(),
            true_field=true_velocity_next[:, 0].detach().cpu(),
            pred_field=pred_velocity_next[:, 0].detach().cpu(),
            error_field=error_velocity[:, 0].detach().cpu(),
            component_name="u",
            save_path=os.path.join(plot_dir, f"sample_{sample_idx:03d}_u_comparison.png"),
        )

        make_comparison_plot(
            mesh_pos=sample.mesh_pos.detach().cpu(),
            cells=sample.cells.detach().cpu(),
            true_field=true_velocity_next[:, 1].detach().cpu(),
            pred_field=pred_velocity_next[:, 1].detach().cpu(),
            error_field=error_velocity[:, 1].detach().cpu(),
            component_name="v",
            save_path=os.path.join(plot_dir, f"sample_{sample_idx:03d}_v_comparison.png"),
        )

        print(f"Saved plots for sample {sample_idx}")

    print(f"Saved evaluation figures to: {plot_dir}")


if __name__ == "__main__":
    main()