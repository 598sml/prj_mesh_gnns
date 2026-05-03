import os
import pathlib
import numpy as np
import functools
import json
import h5py
import argparse

import tensorflow as tf
tfv1 = tf.compat.v1


def _parse(proto, meta):
    """Parses a trajectory from tf.Example, i.e. decode (map) the data."""
    feature_lists = {k: tfv1.io.VarLenFeature(tfv1.string) for k in meta["field_names"]}
    features = tfv1.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tfv1.io.decode_raw(features[key].values, getattr(tfv1, field["dtype"]))
        data = tfv1.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tfv1.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tfv1.io.decode_raw(features["length_" + key].values, tfv1.int32)
            length = tfv1.reshape(length, [-1])
            data = tfv1.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path: str, split: str):
    """Load dataset and decode (map) it."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tfv1.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def save_numpy_as_hdf5(
    data_dict: dict[str, dict[str, np.array]], fname: str | pathlib.Path
) -> None:
    """Write a dictionary of numpy arrays to an hdf5 file."""
    dir_fname = pathlib.Path(fname).parent
    if not os.path.isdir(dir_fname):
        os.makedirs(dir_fname)

    f = h5py.File(fname + ".hdf5", "w")
    meta = {}
    for grp_name in data_dict:
        meta[grp_name] = {}
        grp = f.create_group(str(grp_name))
        for dset_name in data_dict[grp_name]:
            arr = data_dict[grp_name][dset_name]
            meta[grp_name][dset_name] = str(type(arr)) + " " + str(arr.dtype)
            grp.create_dataset(dset_name, data=arr)
    f.close()

    with open(fname + ".json", "w") as f:
        json.dump(obj=meta, fp=f, indent=4)

    print("Saved", len(data_dict.keys()), "trajectories to", fname + ".hdf5")


def slice_trajectory_time_window(traj_dict, t_start: int, window_len: int):
    """
    Slice all time-dependent arrays in one trajectory dictionary.
    We assume the first axis is time for the cylinder_flow fields.
    """
    t_end = t_start + window_len
    out = {}

    for key, arr in traj_dict.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{key} is not a numpy array")

        if arr.shape[0] < t_end:
            raise ValueError(
                f"Requested time window [{t_start}:{t_end}] for field '{key}', "
                f"but array only has shape {arr.shape}"
            )

        out[key] = arr[t_start:t_end]

    return out


if __name__ == "__main__":
    print("-" * 80, "\nStart", __file__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-in",
        "--indir",
        dest="indir",
        default="data/raw/tfrecord/train",
        help=".tfrecord file prefix to load (without extension is also fine)",
        type=str,
    )
    parser.add_argument(
        "-out",
        "--outdir",
        dest="outdir",
        default="data/raw/hdf5_windowed/train_25traj_200graphs",
        help="output hdf5 file prefix (without extension)",
        type=str,
    )
    parser.add_argument(
        "--traj_start",
        dest="traj_start",
        default=0,
        help="starting trajectory index",
        type=int,
    )
    parser.add_argument(
        "--num_traj",
        dest="num_traj",
        default=1,
        help="number of trajectories to save",
        type=int,
    )
    parser.add_argument(
        "--t_start",
        dest="t_start",
        default=0,
        help="starting raw time-state index",
        type=int,
    )
    parser.add_argument(
        "--window_len",
        dest="window_len",
        default=201,
        help="number of raw time states to keep; 201 -> 200 graph samples after hdf5_to_pyg",
        type=int,
    )

    args = parser.parse_args()

    dataset_dir = pathlib.Path(__file__).parent
    data_dir = dataset_dir.parent
    root_dir = data_dir.parent

    indir = os.path.splitext(args.indir)[0]

    if os.path.exists(f"{root_dir}/{indir}.tfrecord"):
        file_dir = root_dir
    elif os.path.exists(f"{data_dir}/{indir}.tfrecord"):
        file_dir = data_dir
    elif os.path.exists(f"{dataset_dir}/{indir}.tfrecord"):
        file_dir = dataset_dir
    else:
        raise FileNotFoundError(f"Could not find file {root_dir}/{indir}.tfrecord")

    folder_in = pathlib.Path(indir).parent
    split_name = pathlib.Path(indir).stem

    ds = load_dataset(f"{file_dir}/{folder_in}", split_name)

    selected = {}
    traj_stop = args.traj_start + args.num_traj

    for traj_idx, traj in enumerate(ds.as_numpy_iterator()):
        if traj_idx < args.traj_start:
            continue
        if traj_idx >= traj_stop:
            break

        sliced = slice_trajectory_time_window(
            traj_dict=traj,
            t_start=args.t_start,
            window_len=args.window_len,
        )

        new_idx = len(selected)
        selected[str(new_idx)] = sliced

        print(
            f"Selected original trajectory {traj_idx} -> saved as {new_idx}, "
            f"time window [{args.t_start}:{args.t_start + args.window_len}]"
        )

    if len(selected) != args.num_traj:
        raise ValueError(
            f"Requested {args.num_traj} trajectories starting at {args.traj_start}, "
            f"but only collected {len(selected)}"
        )

    save_numpy_as_hdf5(selected, fname=f"{file_dir}/{args.outdir}")

    print("End", __file__)
    print("-" * 80)