import torch
import torch.nn as nn
import sys
import os
import argparse

from src.tools.tools import get_default_device
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.attack.attacker import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='dir to save output file as .pt file')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--val', action='store_true', help='apply attack to validation data')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the test data or validation data
    if not args.val:
        ds = data_sel(args.data_name, args.data_dir_path, train=False)
    else:
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)

    # Load model
    model = model_sel(args.model_name, model_path=args.model_path)
    model.to(device)

    # Get minimum perturbation sizes per sample
    perts = Attacker.get_all_pert_sizes(ds, model, device)
    perts = torch.Tensor(perts)

    # Report mean and standard deviation
    print(f'Mean: {torch.mean(perts)}\tStd: {torch.std(perts)}')

    # Save the perturbation sizes
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}.pt'
    torch.save(perts, out_file)