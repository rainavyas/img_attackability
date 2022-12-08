import torch
from torch.utils.data import Subset, TensorDataset
from sklearn.model_selection import train_test_split

from .data_selector import data_sel


def data_attack_sel(name, root, pert_paths, thresh=0.2, val=0.2, use_val=True, only_correct=False, preds=None, spec=False, vspec=False, unattackable=False):
    '''
    For a single sample:
        if ALL model perturbations are smaller than threshold => attackable -> label 1
        Otherwise -> label 0 (unattackable samples).

        If spec: 
        if mulitple models passed in pert_paths, last model is target. Return attackable sample only if attackable for target, but not universally attackable for all models.

        If vspec:
        if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other models.

        If unattackability is True, then same thing as attackability but flipped.
    '''
    ps = [torch.load(p) for p in pert_paths]

    if unattackable:
        attackability_labels = unattackable_labels(ps, thresh, spec=False, vspec=False)
    else:
        attackability_labels = attackable_labels(ps, thresh, spec=False, vspec=False)
    
    if use_val:
        _, ds = data_sel(name, root, train=True)
    else:
        ds = data_sel(name, root, train=False)
    xs = []
    labels = []
    for i in range(len(ds)):
        x,l = ds[i]
        xs.append(x)
        labels.append(l)


    if only_correct:
        # filter to keep only samples correctly classified by ALL models
        preds = [torch.load(p) for p in preds]
        kept_xs = []
        kept_attackability_labels = []
        for sample in zip(xs, attackability_labels, labels, *preds):
            l = sample[2]
            correct = True
            for pred in sample[3:]:
                pred_ind = torch.argmax(pred).item()
                if pred_ind != l:
                    correct = False
                    break
            if correct:
                kept_xs.append(sample[0])
                kept_attackability_labels.append(sample[1])
        
        xs = kept_xs
        attackability_labels = kept_attackability_labels
        
    xs = torch.stack(xs, dim=0)
    attackability_labels = torch.LongTensor(attackability_labels)
    ds = TensorDataset(xs, attackability_labels)

    if use_val:
        # split into train and validation
        num_val = int(val*len(ds))
        train_indices, val_indices = train_test_split(range(len(ds)), test_size=num_val, random_state=42)
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)

        return train_ds, val_ds
    else:
        return ds

def attackable_labels(ps, thresh, spec=False, vspec=False):
    '''attackable'''
    attackability_labels = []

    for sample in zip(*ps):
        num_attackable = 0
        for pert in sample:
            if pert <= thresh:
                num_attackable += 1

        if spec:
            if num_attackable == len(sample) or sample[-1] > thresh:
                attackability_labels.append(0)
            else:
                attackability_labels.append(1)
        elif vspec:
            if num_attackable == 1 and sample[-1] <= thresh:
                attackability_labels.append(1)
            else:
                attackability_labels.append(0)
        else:
            # find universally attackable samples
            if num_attackable == len(sample):
                attackability_labels.append(1)
            else:
                attackability_labels.append(0)

    return attackability_labels

def unattackable_labels(ps, thresh, spec=False, vspec=False):
    '''unattackable'''
    unattackability_labels = []

    for sample in zip(*ps):
        num_unattackable = 0
        for pert in sample:
            if pert >= thresh:
                num_unattackable += 1

        if spec:
            if num_unattackable == len(sample) or sample[-1] < thresh:
                unattackability_labels.append(0)
            else:
                unattackability_labels.append(1)
        elif vspec:
            if num_unattackable == 1 and sample[-1] >= thresh:
                unattackability_labels.append(1)
            else:
                unattackability_labels.append(0)
        else:
            # find universally unattackable samples
            if num_unattackable == len(sample):
                unattackability_labels.append(1)
            else:
                unattackability_labels.append(0)
    return unattackability_labels