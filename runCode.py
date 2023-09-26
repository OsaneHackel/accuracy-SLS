
from exp_configs import EXP_GROUPS
import trainval


BASE_NAME = "/var/tmp/osane/code/bachelorarbeit/results"
DATA_DIR = "/var/tmp/osane/code/bachelorarbeit/data"

exp_mnist = ['mnist_adam', 'mnist_sgd', 'mnist_sls', 'mnist_sls_acc']
exp_cifar10 = ['cifar10_adam', 'cifar10_sgd', 'cifar10_sls', 'cifar10_sls_acc']
exp_cifar10_dense = ['cifar10_adam_dense', 'cifar10_sgd_dense','cifar10_sls_dense', 'cifar10_sls_acc_dense']
exp_cifar100 = ['cifar100_adam', 'cifar100_sgd', 'cifar100_sls', 'cifar100_sls_acc']
exp_slope=['mnist_slope', 'cifar10_slope', 'cifar100_slope', 'cifar10_slope_dense']
exp_batch=['mnist_batch', 'cifar10_batch', 'cifar100_batch', 'cifar10_batch_dense']


def run_experiments(experiments, results_dir, data_dir=DATA_DIR, replace_results=False, metrics_flag=1, DerivTest=False, epsilon=1e-5, runtime=False, normalization=False, dry_run=False):
    for experiment in experiments:
        for exp_dct in EXP_GROUPS[experiment]:
            if dry_run:
                print(exp_dct)
            else:
                trainval.trainval(exp_dct, results_dir, data_dir,
                                reset=replace_results, metrics_flag=metrics_flag, DerivTest=DerivTest, epsilon=epsilon, runtime=runtime, normalization=normalization)


#Explanation on how to use pathlib.rglob:
# https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob

from pathlib import Path
import pandas as pd
import numpy as np
import json


def mean_val_acc(dataset, model, optimizer):
    accuracies = []
    for json_path in Path('./results').rglob('exp_dict.json'):
        print('for loop')
        model_dir = json_path.parent
        score_list_path = model_dir / 'score_list.pkl'

        with open(json_path, 'r') as f:
            exp_dict = json.load(f)

        exp_model = exp_dict['model']
        exp_optimizer = exp_dict['opt']['name']
        exp_dataset = exp_dict['dataset'] #TODO: delete ['name']?

        if exp_dataset != dataset:
            continue

        if exp_model != model:
            continue

        if exp_optimizer != optimizer:
            continue

        score_list = pd.read_pickle(score_list_path)
        print(score_list)
        accuracies.append(np.max([dct['val_acc'] for dct in score_list]))
        print(accuracies)
    return accuracies

def main():
    #run_experiments(exp_cifar100, f'{BASE_NAME}/new', dry_run=False)
    mean_val_acc('mnist','mlp','adamw')

if __name__ == '__main__':
    main()
