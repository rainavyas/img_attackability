import torch
import torch.nn as nn
import sys
import os
import argparse
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import model_sel
from src.data.data_selector import data_sel
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=200, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--momentum', type=float, default=0.9, help="Specify momentum")
    commandLineParser.add_argument('--weight_decay', type=float, default=1e-4, help="Specify momentum")
    commandLineParser.add_argument('--sch', type=int, default=[100, 150], nargs='+', help="Specify scheduler cycle")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_seed{args.seed}.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data
    train_ds, val_ds = data_sel(args.data_name, args.data_dir_path, train=True)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    # Initialise model
    model = model_sel(args.model_name, num_classes=args.num_classes)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(train_dl, val_dl, out_file, max_epochs=args.epochs)