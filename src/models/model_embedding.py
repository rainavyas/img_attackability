from .model_selector import model_sel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def model_embed(dl, model_name, model_path, device, bs=64, shuffle=False, num_classes=10, bearpaw=False):

    model = model_sel(model_name=model_name, model_path=model_path, num_classes=num_classes, bearpaw=bearpaw)
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
    # import pdb; pdb.set_trace()
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
    
    #### Bearpaw models ####
    elif model_name == 'vgg19':
        return vgg19_embed(model, x)
    elif model_name == 'densenet190':
        return densenet190_embed(model, x)
    elif model_name == 'wrn2810':
        return wrn2810_embed(model, x)
    elif model_name == 'resnext29864':
        return resnext29864_embed(model, x)

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

def densenet190_embed(model, x):
    # assume bearpaw
    x = model.conv1(x)
    x = model.trans1(model.dense1(x)) 
    x = model.trans2(model.dense2(x)) 
    x = model.dense3(x)
    x = model.bn(x)
    x = model.relu(x)
    x = model.avgpool(x)
    return x.squeeze(dim=-1).squeeze(dim=-1)

def wrn2810_embed(model, x):
    # assume bearpaw
    out = model.conv1(x)
    out = model.block1(out)
    out = model.block2(out)
    out = model.block3(out)
    out = model.relu(model.bn1(out))
    return out.squeeze(dim=-1)

def vgg19_embed(model, x):
    # assume bearpaw
    x = model.features(x)
    return x.squeeze(dim=-1).squeeze(dim=-1)

def resnext29864_embed(model, x):
    # assume bearpaw
    x = model.conv_1_3x3.forward(x)
    x = F.relu(model.bn_1.forward(x), inplace=True)
    x = model.stage_1.forward(x)
    x = model.stage_2.forward(x)
    x = model.stage_3.forward(x)
    x = F.avg_pool2d(x, 8, 1)
    return x.squeeze(dim=-1).squeeze(dim=-1)


