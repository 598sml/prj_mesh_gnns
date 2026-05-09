import json
import random
import numpy as np
import torch


def load_json_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
