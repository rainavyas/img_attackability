from .model_selector import model_sel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def model_embed(dl, model_name, model_path, device, bs=64, shuffle=False):

    model = model_sel(model_name=model_name, model_path=model_path)
    model.eval()
    model.to(device)

    all_features = []
    labels = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            features = get_features(model_name, model, x)
            all_features.append(features.cpu())
            labels.append(y)
    X = torch.cat(all_features, dim=0)
    y = torch.cat(labels, dim=0)

    num_feats = X.size(-1)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle), num_feats


def get_features(model_name, model, x):
    if model_name == 'vgg16':
        return vgg16_embed(model, x)
    elif model_name == 'resnet18':
        return resnet18_embed(model, x)
    elif model_name == 'densenet121':
        return densenet121_embed(model, x)

def vgg16_embed(model, x):
    features = model.features(x)
    part_classifier = nn.Sequential(*list(model._classifier.children())[:3])
    return part_classifier(features.squeeze())

def resnet18_embed(model, x):
    features = model.features(x).squeeze(dim=-1)
    return features.squeeze(dim=-1)

def densenet121_embed(model, x):
    features = model.features(x).squeeze(dim=-1)
    return features.squeeze(dim=-1)
