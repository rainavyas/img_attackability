'''
Analyze properties of the data wrt to its (un)attackability

--class_dist:
    Plot the original class distribution for the uni (un)attackable samples
'''

import sys
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from src.data.attackability_data import data_attack_sel

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. cifar10')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--perts', type=str, required=True, nargs='+', help='paths to perturbations')
    commandLineParser.add_argument('--plot_dir', type=str, required=True, help='path to plot directory')
    commandLineParser.add_argument('--thresh', type=float, default=0.2, help="Specify imperceptibility threshold")
    commandLineParser.add_argument('--unattackable', action='store_true', help='detector trained to detect unattackable samples')
    commandLineParser.add_argument('--class_dist', action='store_true', help='view class distribution')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyze.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    if args.class_dist:
        # Visualize the class distribution for the (un)attackable samples

        # Load the attacked test data
        ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, use_val=True, val_for_train=False, unattackable=args.unattackable, ret_labels=True)

        # Get class distribution over positive labels
        class_count_dict = defaultdict(int)
        total = 0
        labels = []
        for i in tqdm(range(len(ds))):
            _, att_lab, lab = ds[i]
            labels.append(lab.item())
            if att_lab.item() == 1:
                class_count_dict[lab.item()] += 1
                total += 1
        num_classes = len(set(labels))


        # Normalize the counts and order in descending order
        class_count_dict = {k:v/total for (k,v) in class_count_dict.items()}
        class_inds = list(class_count_dict.keys())
        class_vals = list(class_count_dict.values())

        class_inds = [str(k) for k,_ in sorted(zip(class_inds, class_vals), reverse=True, key=lambda p: p[1])]
        class_vals = sorted(class_vals, reverse=True)
        
        # Plot in descending order (clearly shows any deviation from uniform class dist)
        out_file = f'{args.plot_dir}/{args.data_name}_class-dist{args.class_dist}_unattackable_{args.unattackable}_thresh{args.thresh}.png'
        sns.set_style('darkgrid')

        sns.barplot(x=class_inds, y=class_vals)
        plt.xlabel('Class Index')
        plt.ylabel('Class Fraction')
        plt.savefig(out_file, bbox_inches='tight')

        # Calculate F1 for detecting (un)attackable samples using largest frac classes
        # select classes that are more than 2 times more than uniform fraction
        uni_frac = 1/num_classes
        pos_inds = [k for k,v in class_count_dict.items() if v>uni_frac*2]

        targets = []
        preds = []
        for i in tqdm(range(len(ds))):
            _, att_lab, lab = ds[i]
            targets.append(att_lab.item())
            if lab.item() in pos_inds:
                preds.append(1)
            else:
                preds.append(0)
        
        _, _, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
        print("Dominant classes F1: ", f1)