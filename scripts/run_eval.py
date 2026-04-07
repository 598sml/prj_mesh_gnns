import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from torch_geometric.loader import DataLoader

from meshgraphnet import MeshGraphNet
from meshgraphnet.train_eval import evaluate

def main():
    checkpoint_path = "outputs/checkpoints/meshgraphnet_first_run.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    cfg = checkpoint["cfg"]
    stats_list = checkpoint["stats_list"]
    num_node_features = checkpoint["num_node_features"]
    num_edge_features = checkpoint["num_edge_features"]
    num_classes = checkpoint["num_classes"]

    device = cfg.device

    model = MeshGraphNet(
        input_dim_node = num_node_features,
        input_dim_edge = num_edge_features,
        hidden_dim = cfg.model.hidden_dim,
        output_dim = num_classes,
        cfg = cfg,
    ).to(device)

    # load_state_dict built in function that loads the model parameters from a checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_path = os.path.join("data/processed/data_pt/test.pt")
    data_test = torch.load(test_path, weights_only=False)

    test_loader = DataLoader(
        data_test,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = stats_list
    mean_vec_x = mean_vec_x.to(device)
    std_vec_x = std_vec_x.to(device)
    mean_vec_edge = mean_vec_edge.to(device)
    std_vec_edge = std_vec_edge.to(device)
    mean_vec_y = mean_vec_y.to(device)
    std_vec_y = std_vec_y.to(device)

    test_loss, velocity_rmse = evaluate(
        loader=test_loader,
        device=device,
        model=model,
        mean_vec_x=mean_vec_x,
        std_vec_x=std_vec_x,
        mean_vec_edge=mean_vec_edge,
        std_vec_edge=std_vec_edge,
        mean_vec_y=mean_vec_y,
        std_vec_y=std_vec_y,
    )

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Velocity RMSE: {velocity_rmse:.6f}")

if __name__ == "__main__":
    main()