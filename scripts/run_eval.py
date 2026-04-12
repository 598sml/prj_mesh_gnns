import os
import sys
import random
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from meshgraphnet.config import Config
from meshgraphnet.model import MeshGraphNet
from meshgraphnet.train_eval import evaluate
from meshgraphnet import normalization as norm


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rebuild_cfg_from_dict(cfg_dict):
    """
    Rebuild a Config object from the plain dictionary saved in the checkpoint.
    """
    cfg = Config()

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.model.hidden_dim = cfg_dict["model"]["hidden_dim"]
    cfg.model.num_layers = cfg_dict["model"]["num_layers"]

    cfg.training.batch_size = cfg_dict["training"]["batch_size"]
    cfg.training.learning_rate = cfg_dict["training"]["learning_rate"]
    cfg.training.weight_decay = cfg_dict["training"]["weight_decay"]
    cfg.training.num_epochs = cfg_dict["training"]["num_epochs"]

    if not hasattr(cfg, "data"):
        cfg.data = SimpleNamespace()

    cfg.data.noise_scale = cfg_dict["data"]["noise_scale"]
    cfg.data.noise_gamma = cfg_dict["data"]["noise_gamma"]

    return cfg


def check_consecutive_pair(test_data, delta_t, i=0):
    """
    Check whether graph i and graph i+1 are consecutive time steps on the same mesh.
    """
    if i + 1 >= len(test_data):
        print(f"Cannot check pair {i} -> {i+1}: not enough samples.")
        return

    g0 = test_data[i]
    g1 = test_data[i + 1]

    print(f"\nConsistency check for pair {i} -> {i+1}")
    print("same edge_index:", torch.equal(g0.edge_index, g1.edge_index))
    print("same cells:", torch.equal(g0.cells, g1.cells))
    print("same mesh_pos:", torch.allclose(g0.mesh_pos, g1.mesh_pos))

    v1_from_g0 = g0.x[:, 0:2] + g0.y * delta_t
    v1_actual = g1.x[:, 0:2]

    print("allclose:", torch.allclose(v1_from_g0, v1_actual, atol=1e-6, rtol=1e-5))
    print("max abs diff:", torch.max(torch.abs(v1_from_g0 - v1_actual)).item())
    print("mean abs diff:", torch.mean(torch.abs(v1_from_g0 - v1_actual)).item())


def make_comparison_plot(mesh_pos, cells, true_field, pred_field, error_field, component_name, save_path):
    """
    Save a 3-panel comparison: truth, prediction, error.
    """
    triang = mtri.Triangulation(
        mesh_pos[:, 0].cpu().numpy(),
        mesh_pos[:, 1].cpu().numpy(),
        cells.cpu().numpy(),
    )

    true_np = true_field.cpu().numpy()
    pred_np = pred_field.cpu().numpy()
    err_np = error_field.cpu().numpy()

    vmin = min(true_np.min(), pred_np.min())
    vmax = max(true_np.max(), pred_np.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    im0 = axes[0].tripcolor(triang, true_np, shading="flat", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True {component_name}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].tripcolor(triang, pred_np, shading="flat", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted {component_name}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].tripcolor(triang, err_np, shading="flat")
    axes[2].set_title(f"Error {component_name}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    set_seed(5)
    delta_t = 0.01

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_path = os.path.join(
        base_dir, "outputs", "checkpoints", "meshgraphnet_first_run.pt"
    )
    test_data_path = os.path.join(base_dir, "test_ordered_debug.pt")

    print("checkpoint_path:", checkpoint_path)
    print("test_data_path:", test_data_path)
    print("checkpoint exists:", os.path.exists(checkpoint_path))
    print("test exists:", os.path.exists(test_data_path))

    # ---------- Load checkpoint ----------
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    cfg_dict = checkpoint["cfg_dict"]
    stats_list = checkpoint["stats_list"]

    cfg = rebuild_cfg_from_dict(cfg_dict)
    print(f"Using device: {cfg.device}")

    # ---------- Load test data ----------
    test_data = torch.load(test_data_path, weights_only=False)
    print(f"Loaded {len(test_data)} test graph samples.")

    # ---------- Dataset consistency check ----------
    check_consecutive_pair(test_data, delta_t, i=0)

    # ---------- Rebuild model ----------
    model = MeshGraphNet(
        input_dim_node=checkpoint["num_node_features"],
        input_dim_edge=checkpoint["num_edge_features"],
        hidden_dim=cfg.model.hidden_dim,
        output_dim=checkpoint["num_classes"],
        cfg=cfg,
    ).to(cfg.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---------- Move stats to device ----------
    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = stats_list

    mean_vec_x = mean_vec_x.to(cfg.device)
    std_vec_x = std_vec_x.to(cfg.device)
    mean_vec_edge = mean_vec_edge.to(cfg.device)
    std_vec_edge = std_vec_edge.to(cfg.device)
    mean_vec_y = mean_vec_y.to(cfg.device)
    std_vec_y = std_vec_y.to(cfg.device)

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

    # ---------- Single-sample visualization ----------
    sample_idx = 0
    sample = test_data[sample_idx].to(cfg.device)

    with torch.no_grad():
        pred = model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

    pred_delta = norm.unnormalize(pred, mean_vec_y, std_vec_y)
    pred_velocity_next = sample.x[:, 0:2] + pred_delta * delta_t
    true_velocity_next = sample.x[:, 0:2] + sample.y * delta_t
    error_velocity = pred_velocity_next - true_velocity_next

    plot_dir = os.path.join(base_dir, "outputs", "eval_figures")
    os.makedirs(plot_dir, exist_ok=True)

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

    print(f"Saved evaluation figures to: {plot_dir}")


if __name__ == "__main__":
    main()