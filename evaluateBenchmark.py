from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tueplots import axes, bundles
import seaborn as sns

RESULT_PATH = '/var/tmp/osane/code/bachelorarbeit/results/benchmark/'

#FIGURE_PATH = '/var/tmp/osane/code/bachelorarbeit/figures/benchmark'

def main():
    parser = ArgumentParser()
    parser.add_argument('--bench', type=str,
                        help='the benchmark to evaluate')
    args = parser.parse_args()
    if args.bench is not None:
        bench = args.bench
        highest_acc = []
        lowest_loss = []
        for i in range(0, 5):
            path = RESULT_PATH + bench + f'/{i}/score_list.pkl'
            epoch_stats=pd.read_pickle(path)
            #print(data_frames)
            highest_acc.append(np.max([dct['val_acc'] for dct in epoch_stats]))
            lowest_loss.append(np.min([dct['train_loss'] for dct in epoch_stats]))
        print(f'acc: {highest_acc} loss: {lowest_loss}')
        print(f'acc: {bench} mean: {np.mean(highest_acc):4e}  and std: {np.std(highest_acc):4e}')
        print(f'loss: {bench} mean: {np.mean(lowest_loss):4e}  and std: {np.std(lowest_loss):4e}')


if __name__ == '__main__':
    main()