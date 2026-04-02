import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
# from torch_geometric.data import Data # Maybe I dont need this line here CHECK TODO

from . import normalization as norm
from .processor import ProcessorLayer

class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, cfg):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. Built upon Deepmind's 2021 paper.
        The model consists of an encoder, processor and decoder.

        Input_dims: dynamic variables (nodes) and edge features (edges)
        Hidden_dim: dimension of the hidden layers in the processor
        Output_dim: dimension of the output.
        """

        self.num_layers = cfg.model.num_layers

        # Encoder
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        # Processor initialization

        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of layers must be at least 1" # the processor must have at least one layer, 10 were used in the paper. CHECK TODO

        processor_layer = self._build_processor_layer()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))


        # Decoder (only for node features, as we are predicting node dynamics)

        self.decoder = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

    def _build_processor_layer(self):
        """
        Builds a single processor layer, which consists of a message passing layer followed by an MLP.
        The message passing layer updates the edge features based on the node features, and then updates the node features based on the edge features.
        """

        return ProcessorLayer
    
    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        """
        Forward pass through the MeshGraphNet model.
        The input data is normalized using the provided mean and std vectors for both node and edge features.
        """

        x, edge_index, edge_attr, pressure = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.p
        )

        # Normalize node and edge features

        x =  norm.normalize(x, mean_vec_x, std_vec_x)
        edge_attr = norm.normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: Encode node and edge features into latent node and edge embeddings
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Step 2: Process the latent embeddings through multiple processor layers
        for processor in self.processor:
            x, edge_attr = processor(x, edge_index, edge_attr)

        # Step 3: Decode the final node embeddings to produce the output predictions
        out = self.decoder(x)

        return out
    
    def loss(self, pred, inputs, mean_vec_y, std_vec_y):

        # Define  the node types we calculate the loss for.
        # Since our model has 4 node types (inlet, outlet, wall, internal)
        # and the data is was simulated from a prescribed inlet velocity and no-slip 
        # boundary conditions at the walls, we only calculate the loss for the internal
        # and oulet nodes, as the inlet and wall nodes have prescribed dynamics.

        # normal means internal fluid nodes.

        normal = torch.tensor(0)
        outflow = torch.tensor(5)

        # Get the loss mask for the normal and outflow nodes
        loss_mask = torch.logical_or((torch.argmax(inputs.x[:, 2:], dim=1) == normal),
                                     (torch.argmax(inputs.x[:, 2:], dim=1) == outflow))

        # Normalize true values with dataset mean and std
        labels = norm.normalize(inputs.y, mean_vec_y, std_vec_y)

        # Sum of square errors for each node (in here considering all the nodes))
        error = torch.sum((labels - pred) ** 2, axis=1) # compute the error for each node, error is [num_nodes]

        # Root mean squared error for the nodes we are interested in (normal and outflow nodes)
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss
