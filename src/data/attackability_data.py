import torch
from torch.utils.data import TensorDataset, random_split

from .data_selector import data_sel


def data_attack_sel(name, root, pert_paths, thresh=0.2, val=0.2):
    '''
    For a single sample:
        if ALL model perturbations are smaller than threshold => attackable -> label 0
        Otherwise -> label 1 (unattackable samples).
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
            labels.append(0)
        else:
            labels.append(1)
        labels = torch.LongTensor(labels)
    
    ds = data_sel(name, root, train=False)
    xs = []
    for i in range(len(ds)):
        x,_ = ds[i]
        xs.append(x)
    xs = torch.stack(xs, dim=0)

    ds = TensorDataset(xs, labels)

    # split into train and validation
    num_val = int(val*len(ds))
    num_train = len(ds) - num_val
    train_ds, val_ds = random_split(ds, [num_train, num_val])
    return train_ds, val_ds
