from cnn_finetune import make_model
import torch

from .linear import SingleLinear

def model_sel(model_name='vgg16', model_path=None, num_classes=10, pretrained=True, size=32):
    if model_name == 'linear':
        model = SingleLinear(size, num_classes)
    else:
        model = make_model(model_name, num_classes=num_classes, pretrained=pretrained, input_size=(size,size))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model