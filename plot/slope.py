import argparse
from pathlib import Path

import common 
import matplotlib.pyplot as plt
import pandas as pd
import copy
from tueplots import figsizes, fontsizes, fonts
plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())
EXPS=[['mnist', 'mlp'],['cifar10','resnet34'],['cifar10', 'densenet121'],['cifar100', 'resnet34_100']]
def create_table_slopes(result_dir, dataset, model, output_path):
    slopes= [0.1,0.5,0.9,1,2,5,10]
    accs = []
    losses = []
    lrs=[]
    for slope in slopes:  
      df_slope=common.give_score_lists(result_dir, dataset, model, 'Sls', 
                                        version='weak_accuracy',
                                        slope=f'{slope}', epoch=100)
      accs.append(common.give_values(df_slope[0], 'val_acc'))
      losses.append(common.give_values(df_slope[0], 'train_loss'))
      lrs.append(common.give_values(df_slope[0], 'step_size'))
    d = {'slope': slopes, 'acc': accs, 'loss': losses, 'lr': lrs}
    dl = pd.DataFrame(data=d)
    #print(dl)
    plot_lr_for_slopes(dl, dataset, model, output_path)
    plot_acc_for_slopes(dl, dataset, model, output_path)
    if dataset == 'cifar10' and model == 'densenet121':
        plot_lr_acc(dl, output_path)
    df=copy.deepcopy(dl)
    df['slope'] = slopes
    df['acc'] = [max(acc_list) for acc_list in accs]
    df['loss'] = [min(loss_list) for loss_list in losses]
    #plot_max_acc_slopes(df, dataset, model, output_path)
    #df.to_csv(output_path / f'slopes_max_{dataset}_{model}.csv', index=False)
    return df

def plot_lr_acc(dl, output_path):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    epoch = range(0,100)
    fig, ax = plt.subplots(1,2)
    for i, slope_val in enumerate(dl['slope']):
        ax[0].plot(epoch, dl['lr'][i], label=f"sf= {slope_val}", linewidth=0.8)
        ax[1].plot(epoch, dl['acc'][i], label=f"sf= {slope_val}", linewidth=0.8)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Learning rate')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Validation Accuracy')
    ax[0].legend(loc='upper left')
    #fig.legend()
    fig.suptitle(f'Learning Rate and Accuracy for Different Slope Factors on Cifar10 Densenet121')
    fig.savefig(output_path / 'slopeAccLR.pdf')

def plot_lr_for_slopes(dl, dataset, model, output_path):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=1))
    epoch = range(0,100)
    fig, ax = plt.subplots()
    for i, slope_val in enumerate(dl['slope']):
        ax.plot(epoch, dl['lr'][i], label=f"slope {slope_val}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    ax.set_title(f'Learning rate for different Slopes on {dataset}_{model}')
    ax.legend()
    #fig.savefig(output_path / f'slopes_lr_{dataset}_{model}.pdf')

def plot_acc_for_slopes(dl, dataset, model, output_path):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=1))
    epoch = range(0,100)
    fig, ax = plt.subplots()
    for i, slope_val in enumerate(dl['slope']):
        accs=dl['acc'][i]
        ax.plot(epoch[1:], accs[1:], label=f"slope {slope_val}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'Accuracy for different Slopes on {dataset}_{model}')
    ax.legend()
    #fig.savefig(output_path / f'slopes_acc_{dataset}_{model}.pdf')

def plot_max_acc_slopes(df, dataset, model, output_path):
    fig, ax = plt.subplots()
    ax.plot(df['slope'], df['acc'], label="Validation Accuracy", marker='o')
    ax.set_xlabel('Slope')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Maximal Validation Accuracy for Different Slope Factors on {dataset}_{model}')
    ax.legend()
    #fig.savefig(output_path / f'slopes_max_val_acc_{dataset}_{model}.pdf')

def max_acc_slopes(result_dir, output_path):
    plt.rcParams.update(figsizes.neurips2021(nrows=2, ncols=2))
    fig, ax = plt.subplots(2,2)
    print('works')
    for i, (dataset, model) in enumerate(EXPS):
        df=create_table_slopes(result_dir, dataset, model, output_path)
        ax[i//2][i%2].plot(df['slope'], df['acc'], label="Validation Accuracy", marker='o', markersize=3)
        ax[i//2][i%2].set_xlabel('Slope')
        ax[i//2][i%2].set_ylabel('Accuracy')
        ax[i//2][i%2].set_title(f'{dataset} {model}')
    fig.suptitle(f'Maximal Validation Accuracy for Different Slope Factors')
    fig.savefig(output_path / f'slopeSearch.pdf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')
    parser.add_argument('--out-dir', type=Path, default='figures\\final')
    args = parser.parse_args()

    max_acc_slopes(args.result_dir, args.out_dir)
    #create_table_slopes(args.result_dir,'mnist', 'mlp', args.out_dir)
    #create_table_slopes(args.result_dir,'cifar10', 'resnet34', args.out_dir)
    #create_table_slopes(args.result_dir,'cifar10', 'densenet121', args.out_dir)
    #create_table_slopes(args.result_dir,'cifar100', 'resnet34_100', args.out_dir)

if __name__ == '__main__':
    main()