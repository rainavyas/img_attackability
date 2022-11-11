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
    commandLineParser.add_argument('--unattackable', action='store_true', help='pr curve for unattackable sample')
    commandLineParser.add_argument('--only_correct', action='store_true', help='filter to only eval with correctly classified samples')
    commandLineParser.add_argument('--preds', type=str, default='', nargs='+', help='If only_correct, pass paths to saved model predictions')
    commandLineParser.add_argument('--trained_model_paths', type=str, nargs='+', default='', help='paths to trained models for embedding linear classifiers')
    commandLineParser.add_argument('--spec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target, but not universally.')
    commandLineParser.add_argument('--vspec', action='store_true', help='if mulitple models passed in perts, last model is target. Label attackable sample only if attackable for target ONLY - no other model.')
    commandLineParser.add_argument('--pr_save_path', type=str, default='', help='path to save raw pr values for later plotting')
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
    ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, use_val=False, only_correct=args.only_correct, preds=args.preds, spec=args.spec, vspec=args.vspec)
    base_dl = torch.utils.data.DataLoader(ds, batch_size=args.bs)

    dls = []
    num_featss = []
    for mname, mpath in zip(args.model_names, args.trained_model_paths):
        if 'linear' in mname or 'fcn' in mname:
            # Get embeddings per model
            trained_model_name = mname.split('-')[-1]
            dl, num_feats = model_embed(base_dl, trained_model_name, mpath, device, bs=args.bs, shuffle=False)
            dls.append(dl)
            num_featss.append(num_feats)
        else:
            dls.append(base_dl)
            num_featss.append(0)

    # Load models
    models = []
    for mname, mpath, n in zip(args.model_names, args.model_paths, num_featss):
        if 'linear' in mname:
            model = model_sel('linear', model_path=mpath, num_classes=2, size=n)
        elif 'fcn' in mname:
            model = model_sel('fcn', model_path=mpath, num_classes=2, size=n)
        else:
            model = model_sel(mname, model_path=mpath, num_classes=2)
        model.to(device)
        models.append(model)

    # Get ensemble probability predictions
    criterion = nn.CrossEntropyLoss().to(device)
    s = torch.nn.Softmax(dim=1)
    all_probs = []
    for dl, model in zip(dls, models):
        logits, labels = Trainer.eval(dl, model, criterion, device, return_logits=True)   
        probs = s(logits)
        all_probs.append(probs)
        labels = labels.detach().cpu().tolist()
    probs = torch.mean(torch.stack(all_probs), dim=0)[:,1].squeeze(dim=-1).detach().cpu().tolist()

    if args.unattackable:
        probs = [1-p for p in probs]
        labels = [1-l for l in labels]

    # Get precision-recall curves
    precision, recall, _ = precision_recall_curve(labels, probs)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    # plot all the data
    out_file = f'{args.plot_dir}/unattackability_{args.unattackable}_thresh{args.thresh}_{args.model_names[0]}_{args.data_name}.png'
    if args.only_correct:
        out_file = f'{args.plot_dir}/unattackability_{args.unattackable}_thresh{args.thresh}_{args.model_names[0]}_{args.data_name}_only-correct.png'
    sns.set_style("darkgrid")
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file, bbox_inches='tight')

    if args.pr_save_path != '':
        np.savez(args.pr_save_path, precision=np.asarray(precision), recall=np.asarray(recall))

