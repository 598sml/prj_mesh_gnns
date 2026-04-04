from .model import MeshGraphNet
from . import normalization as norm

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


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

    