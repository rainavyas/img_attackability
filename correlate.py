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

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--perts1', type=str, required=True, help='path to first set of perturbation')
    commandLineParser.add_argument('--perts2', type=str, required=True, help='path to first set of perturbation')
    commandLineParser.add_argument('--plot', type=str, required=True, help='file path to plot')
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

    # Scatter plot
    name1 = args.perts1
    name1 = name1.split('/')[-1].split('_')[0]
    name2 = args.perts2
    name2 = name2.split('/')[-1].split('_')[0]
    data = pd.DataFrame.from_dict({name1:p1, name2:p2})
    sns.jointplot(data, x=name1, y=name2, kind='reg')
    # plt.xlabel(name1)
    # plt.ylabel(name2)
    plt.savefig(args.plot, bbox_inches='tight')
    


