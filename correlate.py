'''
Find correlation between adv attack perturbation sizes between models
Or
Find correlation between probability of correct and perturbation size
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.attack.attacker import Attacker
from src.data.data_selector import data_sel

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--compare', action='store_true', help='generate comparison plot between perturbations')
    commandLineParser.add_argument('--binary_sweep', action='store_true', help='binary thresh- attackable or not')
    commandLineParser.add_argument('--plot', type=str, required=True, help='file path to plot')
    commandLineParser.add_argument('--only_correct', action='store_true', help='filter to only consider correctly classified samples')
    commandLineParser.add_argument('--preds', type=str, default='', nargs='+', help='If only_correct, pass paths to saved model predictions')
    commandLineParser.add_argument('--data_name', type=str, default='', help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, default='', help='path to data directory, e.g. data')
    commandLineParser.add_argument('--pert_v_corr', action='store_true', help='correlation between prob of being correct and pert size')
    commandLineParser.add_argument('--pert_v_conf', action='store_true', help='correlation between prob of prediction and pert size')
    commandLineParser.add_argument('--corr_v_conf', action='store_true', help='correlation between prob of prediction and prob of being correct')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/correlate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    ps = [torch.load(p) for p in args.perts]
    names = [n.split('/')[-1].split('_')[0] for n in args.perts]

    if args.only_correct:
        # filter to only keep correct samples - assume we are using validation set for correlation script
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)
        preds = [torch.load(p) for p in args.preds]
        kept_inds = []
        for ind, sample in enumerate(zip(range(len(ds)), *preds)):
            l = ds[sample[0]][1]
            correct = True
            for pred in sample[1:]:
                pred_ind = torch.argmax(pred).item()
                if pred_ind != l:
                    correct = False
                    break
            if correct:
                kept_inds.append(ind)
        kept_ps = [p[kept_inds] for p in ps]
        ps = kept_ps

    if args.compare:

        # Assume only two sets of perturbations passed
        p1 = ps[0].tolist()
        p2 = ps[1].tolist()

        # correlations
        pcc, _ = stats.pearsonr(p1, p2)
        spearman, _ = stats.spearmanr(p1, p2)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        # Scatter plot
        name1 = names[0]
        name2 = names[1]
        data = pd.DataFrame.from_dict({name1:p1, name2:p2})
        sns.jointplot(x = name1, y = name2, kind = "reg", data = data, scatter_kws={'s': 1})
        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

    if args.binary_sweep:
        sns.set_style("darkgrid")

        for name, p in zip(names, ps):
            threshs, frac_attackable = Attacker.attack_frac_sweep(p)
            plt.plot(threshs, frac_attackable, label=name, linestyle='dashed')

        threshs, frac_attackable = Attacker.attack_frac_sweep_all(ps)
        plt.plot(threshs, frac_attackable, label='All')
        plt.ylabel('Fraction attackable')
        plt.xlabel('Imperceptibility Threshold')
        plt.legend()

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

    if args.pert_v_corr:
        '''
        Scatter plot of
            probability of being correct 
            vs
            perturbation size
        '''
        ps = ps[0]
        preds = [torch.load(p) for p in args.preds][0]

        # Get probability of correct label - assume validation data is being used
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)
        probs = []
        corrects = []
        for pred, ind in zip(preds, range(len(ds))):
            l = ds[ind][1]
            prob = pred[l].item()
            probs.append(prob)

            correct = torch.argmax(pred).item() == l
            corrects.append(correct)

        pcc, _ = stats.pearsonr(ps, probs)
        spearman, _ = stats.spearmanr(ps, probs)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        sns.set_style("darkgrid")
        name1 = 'Perturbation Size'
        name2 = 'Probability Correct'
        data = pd.DataFrame.from_dict({name1:ps, name2:probs, 'correct':corrects})
        # sns.jointplot(x = name1, y = name2, kind = "reg", data = data, scatter_kws={'s': 1})
        sns.jointplot(x = name1, y = name2, kind='scatter', data = data, hue='correct', s=2, joint_kws={'alpha':0.5})

        # plt.scatter(ps, probs, label='All', s=2)
        # plt.ylabel('Probability Correct')
        # plt.xlabel('Perturbation Size')

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

    if args.pert_v_conf:
        '''
        Scatter plot of
            probabaility of predicted class
            vs
            perturbation size
        '''
        ps = ps[0]
        preds = [torch.load(p) for p in args.preds][0]

        # Get probability of predicted label - assume validation data is being used
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)
        probs = []
        corrects = []
        for pred, ind in zip(preds, range(len(ds))):
            l = ds[ind][1]
            pred_class = torch.argmax(pred).item()
            prob = pred[pred_class].item()
            probs.append(prob)

            correct = pred_class == l
            corrects.append(correct)

        pcc, _ = stats.pearsonr(ps, probs)
        spearman, _ = stats.spearmanr(ps, probs)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        sns.set_style("darkgrid")
        name1 = 'Perturbation Size'
        name2 = 'Prediction Confidence'
        data = pd.DataFrame.from_dict({name1:ps, name2:probs, 'correct':corrects})
        # sns.jointplot(x = name1, y = name2, kind = "reg", data = data, scatter_kws={'s': 1})
        sns.jointplot(x = name1, y = name2, kind='scatter', data = data, hue='correct', s=2, joint_kws={'alpha':0.5})

        # plt.scatter(ps, probs, label='All', s=2)
        # plt.ylabel('Probability Correct')
        # plt.xlabel('Perturbation Size')

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

if args.corr_v_conf:
        '''
        Scatter plot of
            probabaility of predicted class
            vs
            probability of correct class
        '''
        preds = [torch.load(p) for p in args.preds][0]

        # Get probability of correct and predicted label - assume validation data is being used
        _, ds = data_sel(args.data_name, args.data_dir_path, train=True)
        prob_corrs = []
        prob_preds = []
        corrects = []
        for pred, ind in zip(preds, range(len(ds))):
            l = ds[ind][1]
            pred_class = torch.argmax(pred).item()
            prob_preds.append(pred[pred_class].item())
            prob_corrs.append(pred[l].item())

            correct = pred_class == l
            corrects.append(correct)

        pcc, _ = stats.pearsonr(prob_corrs, prob_preds)
        spearman, _ = stats.spearmanr(prob_corrs, prob_preds)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        sns.set_style("darkgrid")
        name1 = 'Probability Correct'
        name2 = 'Prediction Confidence'
        data = pd.DataFrame.from_dict({name1:prob_corrs, name2:prob_preds, 'correct':corrects})
        # sns.jointplot(x = name1, y = name2, kind = "reg", data = data, scatter_kws={'s': 1})
        sns.jointplot(x = name1, y = name2, kind='scatter', data = data, hue='correct', s=2, joint_kws={'alpha':0.5})

        # plt.scatter(ps, probs, label='All', s=2)
        # plt.ylabel('Probability Correct')
        # plt.xlabel('Perturbation Size')

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()



    

    


