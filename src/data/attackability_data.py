import torch
from torch.utils.data import Subset, TensorDataset
from sklearn.model_selection import train_test_split

from .data_selector import data_sel


def data_attack_sel(name, root, pert_paths, thresh=0.2, val=0.2, use_val=True):
    '''
    For a single sample:
        if ALL model perturbations are smaller than threshold => attackable -> label 1
        Otherwise -> label 0 (unattackable samples).
    '''
    ps = [torch.load(p) for p in pert_paths]

    labels = []
    for sample in zip(*ps):
        smaller = True
        for pert in sample:
            if pert > thresh:
                smaller = False
                break
        if smaller:
            labels.append(1)
        else:
            labels.append(0)
    labels = torch.LongTensor(labels)
    
    if use_val:
        _, ds = data_sel(name, root, train=True)
    else:
        ds = data_sel(name, root, train=False)
    xs = []
    for i in range(len(ds)):
        x,_ = ds[i]
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    ds = TensorDataset(xs, labels)

    if use_val:
        # split into train and validation
        num_val = int(val*len(ds))
        train_indices, val_indices = train_test_split(range(len(ds)), test_size=num_val, random_state=42)
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)

    return train_ds, val_ds
