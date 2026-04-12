import torch

train = torch.load("data/processed/data_pt/train.pt", weights_only=False)
test  = torch.load("data/processed/data_pt/test.pt",  weights_only=False)
valid = torch.load("data/processed/data_pt/valid.pt", weights_only=False)

print("len(train) =", len(train))
print("len(test)  =", len(test))
print("len(valid) =", len(valid))

print(train[0])
print(test[0])
print(valid[0])

def summarize(name, dataset):
    x_shapes = sorted(set(tuple(g.x.shape) for g in dataset))
    e_shapes = sorted(set(tuple(g.edge_index.shape) for g in dataset))
    c_shapes = sorted(set(tuple(g.cells.shape) for g in dataset))
    print(name)
    print("  unique x shapes:", x_shapes[:5], "count =", len(x_shapes))
    print("  unique edge_index shapes:", e_shapes[:5], "count =", len(e_shapes))
    print("  unique cells shapes:", c_shapes[:5], "count =", len(c_shapes))

summarize("train", train)
summarize("test", test)
summarize("valid", valid)