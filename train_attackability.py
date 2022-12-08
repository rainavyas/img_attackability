'''
Train system to detect attackable samples

Note, model_name: linear-vgg16 means we train a linear classifier on top of vgg16 embedding layer
                  fcn-vgg16 means we train a fully connected classifier on top of vgg16 embedding layer
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import model_sel
from src.data.attackability_data import data_attack_sel
from src.training.trainer import Trainer
from src.models.model_embedding import model_embed

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=200, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify weight decay")
    commandLineParser.add_argument('--sch', type=int, default=[100, 150], nargs='+', help="Specify scheduler cycle")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--only_correct', action='store_true', help='filter to only train with correctly classified samples')
    commandLineParser.add_argument('--unattackable', action='store_true', help='train to identify unattackable samples')
    commandLineParser.add_argument('--preds', type=str, default='', nargs='+', help='If only_correct, pass paths to saved model predictions')
    commandLineParser.add_argument('--trained_model_path', type=str, default='', help='path to trained model for embedding linear classifier')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data for trained_model")
    commandLineParser.add_argument('--bearpaw', action='store_true', help='use bearpaw model configuration for the trained_model')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/attackability_thresh{args.thresh}_{args.model_name}_{args.data_name}_seed{args.seed}.th'
    if args.only_correct:
        out_file = f'{args.out_dir}/attackability_thresh{args.thresh}_{args.model_name}_{args.data_name}_only-correct_seed{args.seed}.th'


    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_attackability.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data
    train_ds, val_ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, only_correct=args.only_correct, preds=args.preds, unattackable=args.unattackable)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)
    if 'linear' in args.model_name or 'fcn' in args.model_name:
        # Get embeddings
        trained_model_name = args.model_name.split('-')[-1]
        train_dl, num_feats = model_embed(train_dl, trained_model_name, args.trained_model_path, device, bs=args.bs, shuffle=True, num_classes=args.num_classes, bearpaw=args.bearpaw)
        val_dl, _ = model_embed(val_dl, trained_model_name, args.trained_model_path, device, bs=args.bs, shuffle=False, num_classes=args.num_classes, bearpaw=args.bearpaw)

    # Initialise model
    if 'linear' in args.model_name:
        model = model_sel('linear', num_classes=2, size=num_feats)
    elif 'fcn' in args.model_name:
        model = model_sel('fcn', num_classes=2, size=num_feats)
    else:
        model = model_sel(args.model_name, num_classes=2)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(train_dl, val_dl, out_file, max_epochs=args.epochs)