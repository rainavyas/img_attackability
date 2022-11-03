'''
Can pre-defined measures be used to identify attackable samples
generate pr curve
'''
import torch
import torch.nn as nn
import sys
import os
import argparse
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--measure', type=str, required=True, help='e.g. confidence')
    commandLineParser.add_argument('--model_name', type=str, required=True, help=' target model of attack e.g. vgg16')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--perts', type=str, required=True, help='path to perturbations')
    commandLineParser.add_argument('--preds', type=str, required=True, help='path to saved model predictions')
    commandLineParser.add_argument('--plot_dir', type=str, required=True, help='path to plot directory')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pr_measure.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the attacked test data labels
    ds = data_attack_sel(args.data_name, args.data_dir_path, [args.perts], thresh=args.thresh, use_val=False)
    labels = []
    for i in range(len(ds)):
        _,l = ds[i]
        labels.append(l.item())
    
    # load the value as per measure

    if args.measure == 'confidence':
        # use negative confidence (high negative confidence correlates with attackability)
        preds = torch.load(args.preds)
        measure = []
        for p in preds:
            conf = p[torch.argmax(p)].item()
            measure.append(-1*conf)

    # plot pr curve
    precision, recall, _ = precision_recall_curve(labels, measure)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print('Best F1', best_f1)

    filename = f'{args.plot_dir}/{args.measure}_thresh{args.thresh}_{args.model_name}_{args.data_name}.png'
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(out_file, bbox_inches='tight')


