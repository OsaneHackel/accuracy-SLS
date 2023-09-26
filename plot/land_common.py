from pathlib import Path
import json
import pandas as pd
import numpy as np

RESULT_PATH = 'C:\\Users\\osane\\.vscode\\bachelorarbeit\\results\\'

EXPERIMENTS = ['mnist\\sgd0.1',
               'mnist\\sgd0.01',
               'mnist\\Armijo',
               'cifar10\\sgd0.1',
               'cifar10\\sgd0.01',
               'cifar100\\sgd0.1',
               'cifar100\\sgd0.01',
               'cifar100\\allcnnc'
               ]

# returns list of lr and loss or accuracy values for the lr
def give_lr_and_values(data, opt, batch = 'train1', version='acc', step=1953):
    file = RESULT_PATH + f'{data}\\{opt}\\loss_list.pkl'
    df = pd.read_pickle(file)
    lr= []
    values = []
    epoch = 0
    max_epoch = 0
    for dct in df:
        if dct['number_of_steps'] == step:
            lr.append(dct['lr'])
            values.append(dct[f'{batch}_{version}'])
            epoch = dct['epoch']
            max_epoch = dct['max_epoch']
    df2 = pd.DataFrame({'lr':lr, 'values':values})
    return df2, epoch, max_epoch


def main():
    pass

if __name__ == '__main__':
    main()