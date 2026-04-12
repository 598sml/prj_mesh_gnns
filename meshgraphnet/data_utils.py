import os
import random
import torch


def load_graph_list(file_path):
    """
    Load a list of PyG Data graphs from a .pt file.
    """
    return torch.load(file_path, weights_only=False)


def save_graph_list(data_list, file_path):
    """
    Save a list of PyG Data graphs to a .pt file.
    """
    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data_list, file_path)


def slice_dataset(data_list, start, size):
    """
    Return a contiguous slice from a graph list.
    """
    return data_list[start:start + size]


def shuffled_split(data_list, train_size, valid_size, seed=None):
    """
    Shuffle a copy of the graph list, then split into train and valid sets.
    """
    data_copy = list(data_list)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(data_copy)
    else:
        random.shuffle(data_copy)

    data_train = data_copy[:train_size]
    data_valid = data_copy[train_size:train_size + valid_size]

    return data_train, data_valid


def build_ordered_test_data(file_path, test_start, test_size, save_path=None):
    """
    Load the full dataset, extract an ordered contiguous test slice,
    and optionally save it.
    """
    data_all = load_graph_list(file_path)
    test_data = slice_dataset(data_all, test_start, test_size)

    if save_path is not None:
        save_graph_list(test_data, save_path)

    return test_data


def same_mesh(g0, g1):
    """
    Check whether two graph samples share the same mesh/topology.
    """
    if g0.x.shape[0] != g1.x.shape[0]:
        return False
    if g0.edge_index.shape != g1.edge_index.shape:
        return False
    if g0.cells.shape != g1.cells.shape:
        return False
    if g0.mesh_pos.shape != g1.mesh_pos.shape:
        return False

    return (
        torch.equal(g0.edge_index, g1.edge_index)
        and torch.equal(g0.cells, g1.cells)
        and torch.allclose(g0.mesh_pos, g1.mesh_pos)
    )


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


def summarize_graph_list(data_list, max_items=10):
    """
    Print basic shape information for the first few graph samples.
    """
    print(f"num graph samples: {len(data_list)}")
    for i, g in enumerate(data_list[:max_items]):
        print(
            i,
            g.x.shape,
            g.edge_index.shape,
            g.cells.shape,
            g.mesh_pos.shape,
        )