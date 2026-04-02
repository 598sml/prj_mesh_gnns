import torch
import torch_scatter
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing

class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        Processor layer for the MeshGraphNet model. Consists of a message passing layer followed by an MLP.
        The message passing layer updates the edge features based on the node features, and then updates the node features based on the edge features.
        in_channels: dimension of the input node and edge features [128] in the paper
        out_channels: dimension of the output node and edge features [128] in the paper
        """

        # MLP for updating edge features
        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

        # MLP for updating node features
        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

        # Initialize wieghts for the MLPs
        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        for m in self.edge_mlp:
            if isinstance(m, Linear):
                m.reset_parameters()
        for m in self.node_mlp:
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Handle the pre and post-processing of node embeddings,
        as well as initiates message passing by calling the propagate function.

        Message passing and aggregation is handled by the propagate function.

        The update of x has shape [num_nodes, hidden_dim]
        edge_index has shape [2, num_edges]
        edge_attr has shape [num_edges, hidden_dim]
        """

        out, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) # this method is built in the MessagePassing class, it handles the message passing and aggregation. It calls the message and update functions defined below.

        updated_nodes = self.node_mlp(torch.cat([x, out], dim=1)) # concatenate the original node features with the aggregated messages and pass through the node MLP to get the updated node features
        # here dim=1 because we want to concatenate along the feature/embedding dimension, which is the second dimension (the first dimension is the number of nodes)

        updated_nodes = x + updated_nodes # residual connection for the node features

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages for each edge based on the features of the source and target nodes and the edge features.
        x_i: features of the target node (shape [num_edges, in_channels])
        x_j: features of the source node (shape [num_edges, in_channels])
        edge_attr (target edge): features of the edge (shape [num_edges, out_channels])

        The message are the raw embeddings; they are not processed.
        """

        updated_edges = torch.cat([x_i, x_j, edge_attr], dim = 1) # temporary embeddings for the edges.
        updated_edges = edge_attr + self.edge_mlp(updated_edges) # residual connection for the edge features

        return updated_edges
    
    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        Aggregate messages for each node by summing the messages from the incoming edges.
        inputs (updated_edges): messages for each edge (shape [num_edges, out_channels])
        index (edge_index): target node indices for each edge (shape [num_edges])
        dim_size: number of nodes (optional)

        The aggregation is a simple sum of the messages from the incoming edges for each node.
        """

        # The axis along which to index is 0, we need to be consistent with the shape of the inputs and the index. The inputs have shape [num_edges, out_channels], so we want to index along the first dimension (num_edges) to aggregate messages for each node.
        node_dim = 0

        # scatter takes many values and collects them into groups based on an index.
        # here, we are summing the updated_edges for each node based on the index of the target node (edge_index[0,:]) for each edge. The result is a tensor of shape [num_nodes, out_channels] where each row corresponds to the aggregated messages for that node.
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=node_dim, reduce='sum')

        return out, updated_edges