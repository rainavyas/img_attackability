'''
Train on adversarial examples
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from src.tools.tools import get_default_device, set_seeds
from src.tools.attackability_tools import select_attackable_samples
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer


if __name__ == "__main__":

    # Get main command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model adv trained model')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='e.g. path to trained model to do further adv train on')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=10, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify momentum")
    commandLineParser.add_argument('--sch', type=int, default=[100, 150], nargs='+', help="Specify scheduler cycle")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--bearpaw', action='store_true', help='use bearpaw model configuration')
    args = commandLineParser.parse_args()

    # attackable adv training arguments
    attackParser = argparse.ArgumentParser(description='attackable adv training arguments')
    attackParser.add_argument('--rand', action='store_true', help='select samples randomly')
    attackParser.add_argument('--attackable', action='store_true', help='select samples using attackability detector')
    attackParser.add_argument('--frac', type=float, default=0.15, help="fraction of samples to keep")
    attackParser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Specify trained attackability detectors, list if ensemble')
    attackParser.add_argument('--model_names', type=str, nargs='+', required=True, help='e.g. fcn-vgg16, list multiple if ensemble of detectors')
    attackParser.add_argument('--trained_model_paths', type=str, nargs='+', default='', help='paths to trained models for embedding linear classifiers')
    att_args = attackParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_seed{args.seed}.th'
    if att_args.rand or att_args.attackable:
       out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_{args.frac}_seed{args.seed}.th' 

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/adv_train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data
    _, ds = data_sel(args.data_name, args.data_dir_path, train=True)

    # Prune dataset to select some samples if necessary
    if att_args.rand:
        num = int(att_args.frac*len(ds))
        indices = train_test_split(range(len(ds)), test_size=num, random_state=args.seed)
        ds = Subset(ds, indices)
    
    elif att_args.attackable:
        ds = select_attackable_samples(ds, att_args.model_names, att_args.model_paths, att_args.trained_model_paths, device, frac=att_args.frac, bs=args.bs, num_classes=args.num_classes, bearpaw=args.bearpaw)
    

    # Load model
    model = model_sel(args.model_name, model_path=args.model_path, num_classes=args.num_classes, bearpaw=args.bearpaw)
    model.to(device)

    # Obtain adversarial examples from original examples
    ds = Trainer.attack_all(ds, model, device, method='pgd', epsilon=0.03)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True)

    # Define learning objects
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(dl, dl, out_file, max_epochs=args.epochs)