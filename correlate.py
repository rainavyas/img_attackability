'''
Find correlation between adv attack perturbation sizes between models
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


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--compare', action='store_true', help='generate comparison plot between perturbations')
    commandLineParser.add_argument('--binary_sweep', action='store_true', help='binary thresh- attackable or not')
    commandLineParser.add_argument('--plot', type=str, required=True, help='file path to plot')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/correlate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    

    names = [n.split('/')[-1].split('_')[0] for n in args.perts]

    if args.compare:

        # Assume only two sets of perturbations passed
        perts1 = args.perts[0]
        perts2 = args.perts[1]

        p1 = torch.load(perts1).tolist()
        p2 = torch.load(perts2).tolist()

        # correlations
        pcc, _ = stats.pearsonr(p1, p2)
        spearman, _ = stats.spearmanr(p1, p2)
        print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

        # Scatter plot
        name1 = names[0]
        name2 = names[1]
        data = pd.DataFrame.from_dict({name1:p1, name2:p2})
        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

    if args.binarysweep:
        for name, pert in zip(names, args.perts):
            threshs, frac_attackable = Attacker.attack_frac_sweep(pert)
            plt.plot(threshs, frac_attackable, label=name)

        threshs, frac_attackable = Attacker.attack_frac_sweep_all(args.perts)
        plt.plot(threshs, frac_attackable, label='All')
        plt.xlabel('Fraction attackable')
        plt.xlabel('Imperceptibility Threshold')

        plt.savefig(args.plot, bbox_inches='tight')
        plt.clf()

    

    


