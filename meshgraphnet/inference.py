import copy # used for deep copying graphs during rollout
from types import SimpleNamespace

import torch

from .config import Config
from .model import MeshGraphNet
from . import normalization as norm


def rebuild_cfg_from_dict(cfg_dict, device=None):
    """
    Rebuild a Config object from the plain dictionary saved in the checkpoint.
    """
    cfg = Config()

    if device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = device

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


def move_stats_to_device(stats_list, device):
    """
    Move normalization statistics to the requested device.
    """
    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = stats_list

    return (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )


def load_checkpoint_and_model(checkpoint_path, device=None):
    """
    Load checkpoint, rebuild config, rebuild model, load weights,
    and return everything needed for inference/evaluation.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    cfg = rebuild_cfg_from_dict(checkpoint["cfg_dict"], device=device)

    model = MeshGraphNet(
        input_dim_node=checkpoint["num_node_features"],
        input_dim_edge=checkpoint["num_edge_features"],
        hidden_dim=cfg.model.hidden_dim,
        output_dim=checkpoint["num_classes"],
        cfg=cfg,
    ).to(cfg.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = move_stats_to_device(checkpoint["stats_list"], cfg.device)

    return checkpoint, cfg, model, stats


def predict_normalized_increment(model, graph, stats):
    """
    Run the model on one graph and return the predicted normalized target.
    """
    (
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        _mean_vec_y,
        _std_vec_y,
    ) = stats

    with torch.no_grad():
        pred = model(graph, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)

    return pred


def predict_next_velocity(model, graph, stats, delta_t=0.01):
    """
    Predict next-step velocity field from one graph.

    Returns:
        pred_velocity_next: predicted next velocity
        pred_delta: unnormalized predicted increment target
    """
    (
        _mean_vec_x,
        _std_vec_x,
        _mean_vec_edge,
        _std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ) = stats

    pred = predict_normalized_increment(model, graph, stats)
    pred_delta = norm.unnormalize(pred, mean_vec_y, std_vec_y)
    pred_velocity_next = graph.x[:, 0:2] + pred_delta * delta_t

    return pred_velocity_next, pred_delta


def reconstruct_true_next_velocity(graph, delta_t=0.01):
    """
    Reconstruct the true next-step velocity from the graph's stored target y.
    """
    return graph.x[:, 0:2] + graph.y * delta_t


def compute_velocity_rmse(pred_velocity_next, true_velocity_next):
    """
    Compute nodewise velocity RMSE over all nodes.
    """
    error = torch.sum((pred_velocity_next - true_velocity_next) ** 2, dim=1)
    rmse = torch.sqrt(torch.mean(error))
    return rmse.item()


def one_step_pair_rmse(model, graph_t, graph_tp1, stats, delta_t=0.01):
    """
    Predict from graph_t and compare directly to graph_tp1 current velocity.
    """
    pred_velocity_next, _ = predict_next_velocity(
        model=model,
        graph=graph_t,
        stats=stats,
        delta_t=delta_t,
    )

    true_velocity_next = graph_tp1.x[:, 0:2]
    rmse = compute_velocity_rmse(pred_velocity_next, true_velocity_next)

    return rmse, pred_velocity_next, true_velocity_next


def rollout_one_trajectory(model, test_data, stats, delta_t=0.01, device="cpu"):
    """
    Free-running rollout on one ordered same-mesh trajectory.

    Starts from graph 0 current state, predicts graph 1, feeds prediction back in,
    predicts graph 2, etc.

    Returns:
        pred_graphs: list of predicted graphs (including initial graph at index 0)
        rollout_rmse: list of RMSE values for each rollout step
    """
    assert len(test_data) >= 2, "Need at least 2 ordered graphs for rollout."

    current_graph = copy.deepcopy(test_data[0]).to(device)

    pred_graphs = [copy.deepcopy(current_graph).cpu()]
    rollout_rmse = []

    for step in range(len(test_data) - 1):
        true_next_graph = test_data[step + 1].to(device)

        pred_velocity_next, _ = predict_next_velocity(
            model=model,
            graph=current_graph,
            stats=stats,
            delta_t=delta_t,
        )

        true_velocity_next = true_next_graph.x[:, 0:2]
        rmse = compute_velocity_rmse(pred_velocity_next, true_velocity_next)
        rollout_rmse.append(rmse)

        next_graph = copy.deepcopy(current_graph)
        next_graph.x[:, 0:2] = pred_velocity_next

        pred_graphs.append(copy.deepcopy(next_graph).cpu())
        current_graph = next_graph

    return pred_graphs, rollout_rmse