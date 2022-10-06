'''
Find correlation between adv attack perturbation sizes between models
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
import scipy.stats as stats

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--perts1', type=str, required=True, help='path to first set of perturbation')
    commandLineParser.add_argument('--perts2', type=str, required=True, help='path to first set of perturbation')
    commandLineParser.add_argument('--plot', type=str, default='None', help='file path to plot the perturbations against one another')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/correlate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    p1 = torch.load(args.perts1).tolist()
    p2 = torch.load(args.perts2).tolist()

    # correlations
    pcc, _ = stats.pearsonr(p1, p2)
    spearman, _ = stats.spearmanr(p1, p2)
    print(f'PCC:\t{pcc}\nSpearman:\t{spearman}')

    # # Scatter plot
    # if args


