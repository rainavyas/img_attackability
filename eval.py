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
    commandLineParser.add_argument('--model_path_base', type=str, required=True, help='e.g. experiments/trained_models/my_model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--num_seeds', type=int, default=1, help="Specify number of seeds for model to load")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    args = commandLineParser.parse_args()

    # Assume num seeds is one in this script
    model_path = f'{args.model_path_base}1.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the test data
    ds = data_sel(args.data_name, args.data_dir_path, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False)

    # Load model
    model = model_sel(args.model_name, model_path=model_path)
    model.to(device)

    # Evaluate
    criterion = nn.CrossEntropyLoss().to(device)
    acc = Trainer.eval(dl, model, criterion, device)
    print('Accuracy', acc)