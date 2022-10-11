'''
generate pr curve over test data using trained attackability detector (on val)
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

from src.tools.tools import get_default_device, get_best_f_score
from src.models.model_selector import model_sel
from src.data.attackability_data import data_attack_sel
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify trained attackability model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--plot', type=str, required=True, help='file path to plot')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attackability.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the attacked test data
    ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, use_val=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs)

    # Load model
    model = model_sel(args.model_name, model_path=args.model_path, num_classes=2)
    model.to(device)

    # Get probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    logits, labels = Trainer.eval(dl, model, criterion, device, return_logits=True)
    import pdb; pdb.set_trace()
    s = torch.nn.Sigmoid()
    probs = s(logits)

    # Get precision-recall curves
    precision, recall, _ = precision_recall_curve(labels, probs)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    # plot all the data
    sns.set_style("darkgrid")
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(args.plot, bbox_inches='tight')

