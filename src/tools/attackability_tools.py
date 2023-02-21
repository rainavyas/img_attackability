import torch
import torch.nn as nn
import numpy as np

from ..training.trainer import Trainer
from torch.utils.data import Subset, TensorDataset
from ..models.model_embedding import model_embed
from ..models.model_selector import model_sel


def attackability_probs(base_dl, detector_names, detector_paths, embedding_model_paths, device, bs=16, num_classes=10, bearpaw=True, combination='sum'):
    '''
    Ensemble attackability detectors
    '''
    dls = []
    num_featss = []
    for mname, mpath in zip(detector_names, embedding_model_paths):
        if 'linear' in mname or 'fcn' in mname:
            # Get embeddings per model
            trained_model_name = mname.split('-')[-1]
            dl, num_feats = model_embed(base_dl, trained_model_name, mpath, device, bs=bs, shuffle=False, num_classes=num_classes, bearpaw=bearpaw)
            dls.append(dl)
            num_featss.append(num_feats)
        else:
            dls.append(base_dl)
            num_featss.append(0)

    # Load models
    models = []
    for mname, mpath, n in zip(detector_names, detector_paths, num_featss):
        if 'linear' in mname:
            model = model_sel('linear', model_path=mpath, num_classes=2, size=n)
        elif 'fcn' in mname:
            model = model_sel('fcn', model_path=mpath, num_classes=2, size=n)
        else:
            model = model_sel(mname, model_path=mpath, num_classes=2)
        model.to(device)
        models.append(model)

    # Get ensemble probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    s = torch.nn.Softmax(dim=1)
    all_probs = []
    for dl, model in zip(dls, models):
        logits, labels = Trainer.eval(dl, model, criterion, device, return_logits=True)   
        probs = s(logits)
        all_probs.append(probs)
        labels = labels.detach().cpu().tolist()
    if combination == 'sum':
        probs = torch.mean(torch.stack(all_probs), dim=0)[:,1].squeeze(dim=-1).detach().cpu().tolist()
    elif combination == 'product':
        probs = torch.prod(torch.stack(all_probs), dim=0)[:,1].squeeze(dim=-1).detach().cpu().tolist()
    return probs, labels


def select_attackable_samples(ds, detector_names, detector_paths, embedding_model_paths, device, frac=0.15, bs=16, num_classes=10, bearpaw=True):
    '''
    Use a deep attackability detector to select most attackable samples
    '''
    num = int(frac*len(ds))
    # map all labels to 0/1 to simulate binary classification of detector (this is just to make accuracy function work in eval)
    xs = []
    ys = [0]*len(ds)
    for i in range(len(ds)):
        (x, _) = ds[i]
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    labels = torch.LongTensor(ys)
    temp_ds = TensorDataset(xs, labels)

    # get probabilities of being attackable
    base_dl = torch.utils.data.DataLoader(temp_ds, batch_size=bs, shuffle=False)
    probs, _ = attackability_probs(base_dl, detector_names, detector_paths, embedding_model_paths, device, bs=bs, num_classes=num_classes, bearpaw=bearpaw)
    kept_inds = np.argsort(probs)[(-1*num):]
    ds = Subset(ds, kept_inds)
    return ds
