import torch
from .data_selector import data_sel


def data_attack_sel(name, root, pert_paths, thresh=0.2):
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
    
    ds = data_sel(name, root, train=False)
    xs = []
    for i in range(len(ds)):
        x,_ = ds[i]
        print(x.size())
        import pdb; pdb.set_trace()
        xs.append(x)
        