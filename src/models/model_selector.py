from cnn_finetune import make_model
from .bearpaw.vgg import vgg19_bn
from .bearpaw.resnet import resnet
from .bearpaw.wrn import wrn
from .bearpaw.resnext import resnext
from .bearpaw.densenet import densenet
import torch

from .linear import SingleLinear, FCN

def model_sel(model_name='vgg16', model_path=None, num_classes=10, pretrained=True, size=32, bearpaw=False):
    if model_name == 'linear':
        model = SingleLinear(size, num_classes)
    elif model_name == 'fcn':
        model = FCN(size, num_classes)
    elif bearpaw:
        # use bearpaw package
        if model_name == 'vgg19':
            model = vgg19_bn(num_classes=num_classes)
        elif model_name == 'resnet110':
            model = resnet(num_classes=num_classes, depth=110)
        elif model_name == 'wrn2810':
            model = wrn(num_classes=num_classes, depth=28, widen_factor=10)
        elif model_name == 'resnext29864':
            model = resnext(num_classes=num_classes, depth=29, widen_factor=4, cardinality=8)
        elif model_name == 'densenet190':
            model = densenet(num_classes=num_classes, depth=190, growthRate=40)
        elif model_name == 'densenet121':
            model = make_model(model_name, num_classes=num_classes, pretrained=pretrained, input_size=(size,size))
    else:
        # use cnn fine tune package
        model = make_model(model_name, num_classes=num_classes, pretrained=pretrained, input_size=(size,size))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model