'''
generate pr curve over test data using trained attackability detector (on val)
-> can generate pr curve for unattackability if desired
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.tools.tools import get_default_device, get_best_f_score
from src.tools.attackability_tools import attackability_probs
from src.models.model_selector import model_sel
from src.data.attackability_data import data_attack_sel
from src.training.trainer import Trainer
from src.models.model_embedding import model_embed

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Specify trained attackability models, list if ensemble')
    commandLineParser.add_argument('--model_names', type=str, nargs='+', required=True, help='e.g. vgg16, list multiple if ensemble of detectors')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--plot_dir', type=str, required=True, help='path to plot directory')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--bs', type=int, default=64, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--unattackable', action='store_true', help='detector trained to detect unattackable samples')
    commandLineParser.add_argument('--only_correct', action='store_true', help='filter to only eval with correctly classified samples')
    commandLineParser.add_argument('--preds', type=str, default='', nargs='+', help='If only_correct, pass paths to saved model predictions')
    commandLineParser.add_argument('--trained_model_paths', type=str, nargs='+', default='', help='paths to trained models for embedding linear classifiers')
    commandLineParser.add_argument('--num_classes', type=int, default=10, help="Specify number of classes in data for trained_model_paths")
    commandLineParser.add_argument('--bearpaw', action='store_true', help='use bearpaw model configuration for the trained_model')
    commandLineParser.add_argument('--spec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target, but not universally.')
    commandLineParser.add_argument('--vspec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other model.')
    commandLineParser.add_argument('--pr_save_path', type=str, default='', help='path to save raw pr values for later plotting')
    commandLineParser.add_argument('--combination', type=str, default='sum', choices=['sum', 'product'], help="method to combine ensemble of detector probabilities")
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
    ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, use_val=False, val_for_train=False, only_correct=args.only_correct, preds=args.preds, spec=args.spec, vspec=args.vspec, unattackable=args.unattackable)
    base_dl = torch.utils.data.DataLoader(ds, batch_size=args.bs)

    # Get probability predictions from detectors ensembled
    probs, labels = attackability_probs(base_dl, args.model_names, args.model_paths, args.trained_model_paths, device, bs=args.bs, num_classes=args.num_classes, bearpaw=args.bearpaw, combination=args.combination)

    # Get precision-recall curves
    precision, recall, _ = precision_recall_curve(labels, probs)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    # # plot all the data
    # out_file = f'{args.plot_dir}/unattackability_{args.unattackable}_thresh{args.thresh}_{args.model_names[0]}_{args.data_name}.png'
    # if args.only_correct:
    #     out_file = f'{args.plot_dir}/unattackability_{args.unattackable}_thresh{args.thresh}_{args.model_names[0]}_{args.data_name}_only-correct.png'
    # sns.set_style("darkgrid")
    # plt.plot(recall, precision, 'r-')
    # plt.plot(best_recall,best_precision,'bo')
    # plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig(out_file, bbox_inches='tight')

    if args.pr_save_path != '':
        np.savez(args.pr_save_path, precision=np.asarray(precision), recall=np.asarray(recall))

