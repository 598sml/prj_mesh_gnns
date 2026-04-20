from .model import MeshGraphNet
from . import normalization as norm

import torch
from torch_geometric.loader import DataLoader
import time


def train(data_train, data_valid, stats_list, cfg):
    """
    Performs a training loop on the dataset for MeshGraphNet.
    """

    assert (len(data_train) > 0 and len(data_valid) > 0), f"Start training on {len(data_train)} train and {len(data_valid)} valid data points (one time step graph)"


    # torch_geometric DataLoader are used for handling the data of lists of graphs
    # Data is previously shuffled, since we randomly sample time steps from the trajectories, so we do not shuffle here
    train_loader = DataLoader(
        data_train,
        batch_size=cfg.training.batch_size, # number of graph samples in one batch. epoch is one pass through the whole dataset. We define the number of samples in config and with batch size we how many batches per epoch. If batch size is 1, we update the model after every graph sample, if batch size is 10, we update the model after every 10 graph samples. At every batch we are optimizing the weights of the MLPs in the model.
        shuffle=False,  
    )

    valid_loader = DataLoader(
        data_valid,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # Statistics for normalization
    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list

    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (

        mean_vec_x.to(cfg.device),
        std_vec_x.to(cfg.device),
        mean_vec_edge.to(cfg.device),
        std_vec_edge.to(cfg.device),
        mean_vec_y.to(cfg.device),
        std_vec_y.to(cfg.device),
    )

    num_node_features = data_train[0].x.shape[1]
    num_edge_features = data_train[0].edge_attr.shape[1]
    num_classes = 2 # velocity has 2 components.

    model = MeshGraphNet(
        input_dim_node=num_node_features,
        input_dim_edge=num_edge_features,
        hidden_dim = cfg.model.hidden_dim,
        output_dim = num_classes,
        cfg=cfg,
    ).to(cfg.device)  

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    train_losses = []
    valid_losses = []
    velocity_valid_losses = []
    best_valid_loss = float('inf')
    best_model = None

    start_time = time.time()

    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            batch = batch.to(cfg.device)

            optimizer.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        train_losses.append(avg_train_loss)

        # evaluation at every 10 epochs.
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
                best_model = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        else:
            valid_losses.append(valid_losses[-1])
            velocity_valid_losses.append(velocity_valid_losses[-1])

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1}/{cfg.training.num_epochs}, "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Valid Loss: {valid_losses[-1]:.6f}, "
                f"Vel RMSE: {velocity_valid_losses[-1]:.6f}, "
                f"Best Valid: {best_valid_loss:.6f}"
            )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    return (
        model,
        train_losses,
        valid_losses, 
        velocity_valid_losses, 
        best_model, 
        best_valid_loss
    )

def evaluate(
        loader, 
        device,
        model,
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
        delta_t=0.01,
):
    """
    Evaluate one-step prediction loss and reconstructed velocity RMSE
    on a held-out dataset.
    """
    total_loss = 0.0
    num_batches = 0
    total_velocity_rmse = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            batch_loss = model.loss(pred, data, mean_vec_y, std_vec_y)
            total_loss += batch_loss.item()

            # Get the loss mask for the normal and outflow nodes
            node_types = torch.argmax(data.x[:, 2:], dim=1)  # One-hot encoding for node types, here we can define where is the node. Ex nomal, obstacle, outflow, etc
            loss_mask = (node_types == 0) | (node_types == 5)
            
            # evaluate  the velocity (updating velocity)
            eval_velocity = (data.x[:, 0:2] + norm.unnormalize(pred, mean_vec_y, std_vec_y) * delta_t)

            true_velocity = data.x[:, 0:2] + data.y*delta_t

            # evaluate error
            error = torch.sum((eval_velocity - true_velocity)**2, dim=1) # (upred - utrue)^2 + (vpred - vtrue)^2

            # We compute the velocity RMSE only for the normal and outflow nodes, since the inlet and wall nodes have prescribed dynamics and we are not interested in evaluating the velocity prediction at those nodes.
            # We perform the mean over nodes that are normal or outflow, and then we take the square root to get the RMSE in velocity units.
            velocity_rmse = torch.sqrt(torch.mean(error[loss_mask]))
            # Accumulate the total velocity RMSE across batches for later averaging
            total_velocity_rmse += velocity_rmse.item()

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_velocity_rmse = total_velocity_rmse / max(num_batches, 1)

    return avg_loss, avg_velocity_rmse