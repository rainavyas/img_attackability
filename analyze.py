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
        _, ds = data_attack_sel(args.data_name, args.data_dir_path, args.perts, thresh=args.thresh, use_val=True, val=1.0, unattackable=args.unattackable, ret_labels=False)

        # Get class distribution over positive labels
        class_count_dict = defaultdict(int)
        total = 0
        for i in tqdm(range(len(ds))):
            _, att_lab, lab = ds[i]
            if att_lab == 1:
                class_count_dict[lab] += 1
                total += 1


        # Normalize the counts
        class_count_dict = {k:v/total for (k,v) in class_count_dict.items()}
        
        # Plot in descending order by class size (clearly shows any deviation from uniform class dist)