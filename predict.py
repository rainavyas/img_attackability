'''
Save the predictions from a model in a file
- save a torch tensor: num_samples x num_classes
'''

import torch
import torch.nn as nn
import sys
import os
import argparse

from src.tools.tools import get_default_device
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save predictions')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify trained model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--use_val', action='store_true', help='use validation data or test data for predictions')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load data
    if args.use_val:
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)
    else:
        ds = data_sel(args.data_name, args.data_dir_path, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs)

    # Load model
    model = model_sel(args.model_name, model_path=args.model_path, num_classes=args.num_classes)
    model.to(device)

    # Get probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    logits, _ = Trainer.eval(dl, model, criterion, device, return_logits=True)
    s = torch.nn.Softmax(dim=1)
    probs = s(logits)

    # Save
    out_file = f'{args.out_dir}/model_{args.model_name}_data_{args.data_name}.pt'
    torch.save(probs, out_file)


