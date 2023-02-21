'''
Evaluate adversarially trained model:
    1) Accuracy on test data
    2) Adversarial robustness (fooling rate) on test data
'''

import sys
import os
import argparse
import torch
import torch.nn as nn

from src.tools.tools import get_default_device
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer
from src.attack.attacker import Attacker


if __name__ == "__main__":

    # Get main command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='e.g. path to trained model to do further adv train on')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--bearpaw', action='store_true', help='use bearpaw model configuration')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_adv.cmd', 'a') as f:
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
    model = model_sel(args.model_name, model_path=args.model_path, num_classes=args.num_classes, bearpaw=args.bearpaw)
    model.to(device)

    # calculate accuracy
    criterion = nn.CrossEntropyLoss().to(device)
    acc = Trainer.eval(dl, model, criterion, device)
    print('Accuracy (%)', acc)

    # calculate fooling rate
    ds_att = Attacker.attack_all(ds, model, device, method='pgd', epsilon=0.03)
    dl_att = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True)
    acc_att = Trainer.eval(dl_att, model, criterion, device)

    fool = 100*((acc-acc_att)/acc)
    print('Fooling Rate (%)', fool)

