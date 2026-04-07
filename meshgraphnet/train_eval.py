from .model import MeshGraphNet
from . import normalization as norm

import copy
import torch
from torch_geometric.loader import DataLoader


def add_noise(dataset, cfg):
    """
    Add noise to velocity features on NORMAL nodes only, following the repo idea.

    Expected optional config fields:
        cfg.data.noise_scale
        cfg.data.noise_gamma

    If those do not exist or noise_scale <= 0, no noise is applied.
    """
    noise_scale = getattr(getattr(cfg, "data", object()), "noise_scale", 0.0)
    noise_gamma = getattr(getattr(cfg, "data", object()), "noise_gamma", 1.0)

    if noise_scale <= 0.0:
        return dataset

    for datapoint in dataset:
        momentum = datapoint.x[:, :2]
        node_type = datapoint.x[:, 2:]

        noise = torch.empty_like(momentum).normal_(mean=0.0, std=noise_scale)

        # apply noise only to NORMAL nodes (node type 0)
        condition = node_type[:, 0] == torch.ones_like(node_type[:, 0])
        condition = condition.unsqueeze(1).repeat(1, 2)

        noise = torch.where(condition, noise, torch.zeros_like(momentum))

        momentum = momentum + noise
        datapoint.x = torch.cat((momentum, node_type), dim=-1).type(torch.float)

        # repo-style target correction
        datapoint.y = datapoint.y + (1.0 - noise_gamma) * noise

    return dataset


def train(data_train, data_valid, stats_list, cfg):
    """
    Performs a training loop on the dataset for MeshGraphNet.
    Repo-style structure adapted to this codebase.
    """

    # optional noise injection, only if cfg.data.noise_scale exists
    if hasattr(cfg, "data") and getattr(cfg.data, "noise_scale", 0.0) > 0.0:
        data_train = add_noise(data_train, cfg)

    assert (
        len(data_train) > 0 and len(data_valid) > 0
    ), f"Start training on {len(data_train)} train and {len(data_valid)} valid datapoints"

    train_loader = DataLoader(
        data_train,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    valid_loader = DataLoader(
        data_valid,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list

    mean_vec_x = mean_vec_x.to(cfg.device)
    std_vec_x = std_vec_x.to(cfg.device)
    mean_vec_edge = mean_vec_edge.to(cfg.device)
    std_vec_edge = std_vec_edge.to(cfg.device)
    mean_vec_y = mean_vec_y.to(cfg.device)
    std_vec_y = std_vec_y.to(cfg.device)

    num_node_features = data_train[0].x.shape[1]
    num_edge_features = data_valid[0].edge_attr.shape[1]
    num_classes = 2

    model = MeshGraphNet(
        input_dim_node=num_node_features,
        input_dim_edge=num_edge_features,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=num_classes,
        cfg=cfg,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    train_losses = []
    valid_losses = []
    velocity_valid_losses = []

    best_valid_loss = float("inf")
    best_model = None

    for epoch in range(cfg.training.num_epochs):
        total_loss = 0.0
        model.train()
        num_loops = 0

        for batch in train_loader:
            batch = batch.to(cfg.device)

            optimizer.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_loops += 1

        total_loss /= max(num_loops, 1)
        train_losses.append(total_loss)

        # repo-style evaluation every 10 epochs
        if epoch % 10 == 0:
            valid_loss, velocity_valid_rmse = evaluate(
                valid_loader,
                cfg.device,
                model,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
                delta_t=0.01,
            )

            valid_losses.append(valid_loss)
            velocity_valid_losses.append(velocity_valid_rmse)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model.state_dict())

        else:
            valid_losses.append(valid_losses[-1])
            velocity_valid_losses.append(velocity_valid_losses[-1])

        if epoch % 100 == 0:
            print(
                "train loss "
                + str(round(total_loss, 2))
                + " | valid loss "
                + str(round(valid_losses[-1], 2))
                + " | velocity loss "
                + str(round(velocity_valid_losses[-1], 5))
            )

    print("Finished training!")
    print("Min valid set loss:               {0}".format(min(valid_losses)))
    print("Minimum train loss:               {0}".format(min(train_losses)))
    print("Minimum velocity validation loss: {0}".format(min(velocity_valid_losses)))

    return (
        model,
        train_losses,
        valid_losses,
        velocity_valid_losses,
        best_model,
        best_valid_loss,
    )


def evaluate(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    delta_t=0.01,
):
    """
    Calculates held-out loss and velocity RMSE.
    Repo test() adapted and renamed to evaluate().
    """

    loss = 0.0
    velocity_rmse = 0.0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            test_loss = test_model.loss(pred, data, mean_vec_y, std_vec_y)
            loss += test_loss.item()

            node_types = torch.argmax(data.x[:, 2:], dim=1)
            loss_mask = (node_types == 0) | (node_types == 5)

            eval_velocity = (
                data.x[:, 0:2]
                + norm.unnormalize(pred, mean_vec_y, std_vec_y) * delta_t
            )
            true_velocity = data.x[:, 0:2] + data.y * delta_t

            error = torch.sum((eval_velocity - true_velocity) ** 2, dim=1)
            velocity_rmse += torch.sqrt(torch.mean(error[loss_mask])).item()

        num_loops += 1

    return (loss / max(num_loops, 1)), (velocity_rmse / max(num_loops, 1))