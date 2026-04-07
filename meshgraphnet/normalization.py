import torch

def normalize(to_normalize, mean_vec, std_vec):
    """
    Normalizes the input tensor using the provided mean and std vectors.
    The normalization is done by subtracting the mean and dividing by the std for each feature.
    """

    return (to_normalize - mean_vec) / std_vec

def unnormalize(to_unnormalize, mean_vec, std_vec):
    """
    Unnormalizes the input tensor using the provided mean and std vectors.
    The unnormalization is done by multiplying by the std and adding the mean for each feature.
    """

    return to_unnormalize * std_vec + mean_vec


def get_stats(data_list):
    """
    Computes the mean ans standard deviation for node features,
    edge features and node outputs and normalizes them.
    """

    # mean and std for the node features
    # We apply in data_list[0], because all the data in the list have the same shape for x, edge_attr and y. 
    # So it does not matter which element (graph, and by the way there is in total  1200 trajectories* 599 time steps for each = 718800 graph samples)
    # from the list we use to compute it. We employ the first one for simplicity.
    # About x.shape[1:], it is because we want to compute the mean and std for each feature, so we need to keep the feature dimension and ignore the batch dimension (number of graphs in the list).
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    # mean and std for the edge features
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    # mean and std for the node outputs
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    # Define the maximum number of accumulations to perform such that
    # we do not run out of memory.
    max_accumulations = 10**6

    # Define a very small value for normalizing to avoid division by zero.
    epsilon = torch.tensor(1e-8)

    # Define counters for the number of accumulations.
    num_accumulations_x = 0
    num_accumulations_edge = 0
    num_accumulations_y = 0

    # Loop through the data list and accumulate the sum and sum of squares for each feature.
    for data in data_list:
        mean_vec_x += torch.sum(data.x, dim=0)
        std_vec_x += torch.sum(data.x**2, dim=0)
        num_accumulations_x += data.x.shape[0]

        mean_vec_edge += torch.sum(data.edge_attr, dim=0)
        std_vec_edge += torch.sum(data.edge_attr**2, dim=0)
        num_accumulations_edge += data.edge_attr.shape[0]

        mean_vec_y += torch.sum(data.y, dim=0)
        std_vec_y += torch.sum(data.y**2, dim=0)
        num_accumulations_y += data.y.shape[0]

        # If the number of accumulations exceeds the maximum, compute the mean and std and reset the accumulators.
        if(num_accumulations_x > max_accumulations or num_accumulations_edge > max_accumulations or num_accumulations_y > max_accumulations):
            break

    # Compute the mean and std for the node features, edge features and node outputs.
    mean_vec_x /= num_accumulations_x
    # We compute the std using the formula: std = sqrt(E[x^2] - E[x]^2), where E[x^2] is the mean of the squares and E[x] is the mean. We also add a small value epsilon to avoid division by zero.
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accumulations_x - mean_vec_x**2), epsilon)

    # Compute the mean and std for the edge features.
    mean_vec_edge /= num_accumulations_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge / num_accumulations_edge - mean_vec_edge**2), epsilon)

    # Compute the mean and std for the node outputs.
    mean_vec_y /= num_accumulations_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accumulations_y - mean_vec_y**2), epsilon)

    mean_std_dict = [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y]

    return mean_std_dict


