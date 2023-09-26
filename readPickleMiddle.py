import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
#import pickle

RESULT_PATH = '/var/tmp/osane/code/bachelorarbeit/results/cifar10sgd0.1'
FIGURE_PATH = 'figures/cifar10/sgd0.1/middle/'
LEARNING_RATES = [-1, -5e-1, -1e-1, -1e-2, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1]  

ACC_LABELS = {
    'train1_acc': 'actual batch (bs: 128)',
    'train2_acc': 'first batch (bs: 128)',
    'train3_acc': 'second batch (bs: 128)',
    'big_batch_acc': 'big batch (bs: 1000)',
    'val_acc': 'val acc'
}

LOSS_LABELS = {
    'train1_loss': 'actual batch (bs: 128)',
    'train2_loss': 'first batch (bs: 128)',
    'train3_loss': 'second batch (bs: 128)',
    'big_batch_loss': 'big batch (bs: 1000)',
    'val_loss': 'val loss'
}

ALL_LABELS = list(ACC_LABELS.keys()) + list(LOSS_LABELS.keys())

def initialize_arrays(number_of_steps):
    accuracies = {key: np.zeros((len(number_of_steps), len(LEARNING_RATES))) for key in ACC_LABELS}
    losses = {key: np.zeros((len(number_of_steps), len(LEARNING_RATES))) for key in LOSS_LABELS}
    return accuracies, losses

def plot_acc(axes, accuracies):
    for label, (x,y) in accuracies.items():
        axes.plot(x,y,'-o', label=label)

    #axes.legend()
    axes.set_xlabel('Learning rate')
    axes.set_ylabel('Accuracy')
    axes.grid(True, which='both')

def plot_acc_without_val(axes, accuracies):
    for label, (x,y) in accuracies.items():
        #print(label)
        if label == 'val acc':
            #print('skipping val_acc')
            continue
        axes.plot(x,y,'-o', label=label)

    #axes.legend()
    axes.set_xlabel('Learning rate')
    axes.set_ylabel('Accuracy')
    axes.grid(True, which='both')

def plot_loss(axes, losses):
    for label, (x,y) in losses.items():
        axes.plot(x,y,'-o', label=label)

    #axes.legend()
    axes.set_xlabel('Learning rate')
    axes.set_ylabel('Loss')
    axes.grid(True, which='both')


def plot_accuracy_and_loss(d, output_path):
    # Extract relevant data from the data frame
    number_of_steps = np.unique([item['number_of_steps'] for item in d]) # gives the sampeled steps
    #learning_rates = np.arange(-1, 2.2, 0.2)
    
    accuracies, losses = initialize_arrays(number_of_steps)
    epochs = np.zeros((len(number_of_steps)))
    #print(LEARNING_RATES[2:-2])
    #print(accuracies['train1_acc'][1][2:-2])

    for item in d:
        step_idx = np.where(number_of_steps == item['number_of_steps'])[0][0]
        lr_idx = item['lr_idx']
        epochs[step_idx]= item['epoch'] 
        max_epoch = item['max_epoch']
        #batch_size = 128 #only temporary solution 
        #batch_size = item['batch_size'] how I generally want to have it
        for key in ACC_LABELS:
            accuracies[key][step_idx, lr_idx] = item[key]

        for key in LOSS_LABELS:
            losses[key][step_idx, lr_idx] = item[key]

#4 plots in one -> different scales
    # Plot accuracy
    for i, step in enumerate(number_of_steps):
        epoch = epochs[i]

        #plots the accuracy for the different learning rates
        fig, ax = plt.subplots(2,2)

        valuesAcc = {
            ACC_LABELS[key]: (LEARNING_RATES[2:-2], accuracies[key][i][2:-2]) for key in ACC_LABELS
        }
        plot_acc_without_val(ax[0,0], valuesAcc)
        #plot_acc(ax[0,0], valuesAcc)
        ax[0,0].set_title('no log')

        plot_acc_without_val(ax[0,1], valuesAcc)
        #plot_acc(ax[0,1], valuesAcc)
        ax[0,1].set_title('x log')
        ax[0,1].set_xscale('symlog', linthresh=1e-4)

        plot_acc_without_val(ax[1,0], valuesAcc)
        #plot_acc(ax[1,0], valuesAcc)
        ax[1,0].set_title('y log')
        ax[1,0].set_yscale('log')

        plot_acc_without_val(ax[1,1], valuesAcc)
        #plot_acc(ax[1,1],valuesAcc)
        ax[1,1].set_title('x and y log')
        ax[1,1].set_xscale('symlog', linthresh=1e-4)
        ax[1,1].set_yscale('log')

        fig.set_size_inches(15.5, 8.5)
        plt.subplots_adjust(hspace=0.3)
        plt.legend(loc='lower right', bbox_to_anchor=(1.7,1.65), fontsize="16" ) #TODO: change location of legend
        plt.suptitle(f'Middle of Accuracy vs. Learning Rate for Step Number {step} in Epoch {epoch} of {max_epoch}', fontsize=20)
        fig.savefig(output_path + f'middleAccWithoutVal{step}.pdf', dpi=100, bbox_inches='tight') 
        plt.close(fig)


    # Plot loss
        figloss, axloss = plt.subplots(2,2)

        valuesLoss = {
            LOSS_LABELS[key]: (LEARNING_RATES[2:-2], losses[key][i][2:-2]) for key in LOSS_LABELS
        }


        plot_loss(axloss[0,0], valuesLoss)
        axloss[0,0].set_title('no log')

        plot_loss(axloss[0,1], valuesLoss)
        axloss[0,1].set_title('x log')
        axloss[0,1].set_xscale('symlog', linthresh=1e-4)

        plot_loss(axloss[1,0], valuesLoss)
        axloss[1,0].set_title('y log')
        axloss[1,0].set_yscale('symlog', linthresh=1e-4)

        plot_loss(axloss[1,1],valuesLoss)
        axloss[1,1].set_title('x and y log')
        axloss[1,1].set_xscale('symlog', linthresh=1e-4)
        axloss[1,1].set_yscale('symlog', linthresh=1e-4)

        plt.legend(loc='upper right', bbox_to_anchor=(1.7,2.35), fontsize=16) #TODO: change location of legend
        plt.suptitle(f'Loss vs. Learning Rate for Step Number {step} in Epoch {epoch} of {max_epoch}', fontsize=20)
        figloss.set_size_inches(15.5, 8.5)
        plt.subplots_adjust(hspace=0.3)
        #figloss.tight_layout()
        figloss.savefig(output_path + f'middleLoss{step}.pdf', dpi=100, bbox_inches='tight')
        plt.close(figloss)


def plot_middle(result_path, figure_path):
    os.makedirs(figure_path, exist_ok=True)
    file = result_path + f'/loss_list.pkl' 
    d= pd.read_pickle(file)
    plot_accuracy_and_loss(d, figure_path)


def main():
    plot_middle(RESULT_PATH, FIGURE_PATH)
    

if __name__ == '__main__':
    main()