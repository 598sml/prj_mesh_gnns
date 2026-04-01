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

