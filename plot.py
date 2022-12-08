'''
This script is for playing around with plotting
'''

import sys
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.tools.tools import get_best_f_score

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--deep', type=str, required=True, help='deep npz file')
    commandLineParser.add_argument('--conf_matched', type=str, required=True, help='conf npz file for unseen arch')
    commandLineParser.add_argument('--conf_unmatched', type=str, required=True, help='conf npz file for seen archs conf')
    commandLineParser.add_argument('--plot', type=str, required=True, help='path to plot')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    sns.set_style('darkgrid')

    # deep unseen
    npzfile = np.load(args.deep)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    plt.plot(recall, precision, 'b-', label='Deep Unseen')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.3f}", (best_recall,best_precision), fontsize=5)

    # conf unseen
    npzfile = np.load(args.conf_unmatched)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    plt.plot(recall, precision, 'r-', label='Conf Unseen')
    plt.plot(best_recall,best_precision,'ro')
    plt.annotate(F"F1={best_f1:.3f}", (best_recall,best_precision), fontsize=5)

    # conf seen
    npzfile = np.load(args.conf_matched)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    plt.plot(recall, precision, 'g-', label='Conf Seen')
    plt.plot(best_recall,best_precision,'go')
    plt.annotate(F"F1={best_f1:.3f}", (best_recall,best_precision), fontsize=5)


    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(args.plot, bbox_inches='tight')


