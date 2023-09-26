from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
from tueplots import figsizes, fontsizes, fonts
import seaborn as sns
# import readPickleMiddle
# import pickle
from readPickleMiddle import plot_middle

plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())

#sns.set_theme(style="darkgrid")

EXPERIMENT = 'mnist/accSls'

EXPERIMENTS = ['mnist\\sgd0.1',
               'mnist\\sgd0.01',
               'mnist\\Armijo',
               'cifar10\\sgd0.1',
               'cifar10\\sgd0.01',
               'cifar100\\sgd0.1',
               'cifar100\\sgd0.01',
               'cifar100\\allcnnc'
               ]

#RESULT_PATH = '/var/tmp/osane/code/bachelorarbeit/results/'
RESULT_PATH = 'C:\\Users\\osane\\.vscode\\bachelorarbeit\\results\\'

#FIGURE_PATH = '/var/tmp/osane/code/bachelorarbeit/figures/'
FIGURE_PATH = 'C:\\Users\\osane\\.vscode\\bachelorarbeit\\figures\\landscapes\\'

LEARNING_RATES = [-1, -5e-1, -1e-1, -1e-2, -1e-3, -
                  1e-4, 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1]

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
epochs_lr = []
optimal_lr = []
improvement = []


def initialize_arrays(number_of_steps):
    accuracies = {key: np.zeros(
        (len(number_of_steps), len(LEARNING_RATES))) for key in ACC_LABELS}
    losses = {key: np.zeros((len(number_of_steps), len(LEARNING_RATES)))
              for key in LOSS_LABELS}
    return accuracies, losses


def plot_acc(axes, accuracies):
    for label, (x, y) in accuracies.items():
        axes.plot(x, y, '-o', label=label)
        if label == 'val acc':
            annot_max(x, y, axes)

    # axes.legend()
    # annot_max(accuracies, LEARNING_RATES)
    axes.set_xlabel('Learning rate')
    axes.set_ylabel('Accuracy')
    axes.grid(True, which='both')


def plot_loss(axes, losses):
    for label, (x, y) in losses.items():
        axes.plot(x, y, '-o', label=label)

    # axes.legend()
    axes.set_xlabel('Learning rate')
    axes.set_ylabel('Loss')
    axes.grid(True, which='both')


def plot_accuracy_and_loss(d, output_path):
    # Extract relevant data from the data frame
    number_of_steps = np.unique([item['number_of_steps']
                                for item in d])  # gives the sampeled steps
    # learning_rates = np.arange(-1, 2.2, 0.2)

    accuracies, losses = initialize_arrays(number_of_steps)
    epochs = np.zeros((len(number_of_steps)))
    # print(LEARNING_RATES[2:-2])
    # print(accuracies['train1_acc'][1][2:-2])

    for item in d:
        step_idx = np.where(number_of_steps == item['number_of_steps'])[0][0]
        lr_idx = item['lr_idx']
        epochs[step_idx] = item['epoch']

        # probably not the best solution
        epochs_lr.append(item['epoch'])

        max_epoch = item['max_epoch']
        # batch_size = 128 #only temporary solution
        # batch_size = item['batch_size'] how I generally want to have it
        for key in ACC_LABELS:
            accuracies[key][step_idx, lr_idx] = item[key]

        for key in LOSS_LABELS:
            losses[key][step_idx, lr_idx] = item[key]

# 4 plots in one -> different scales
    # Plot accuracy
    for i, step in enumerate(number_of_steps):
        epoch = epochs[i]

        # plots the accuracy for the different learning rates
        fig, ax = plt.subplots(2, 2)

        valuesAcc = {
            ACC_LABELS[key]: (LEARNING_RATES, accuracies[key][i]) for key in ACC_LABELS
        }

        plot_acc(ax[0, 0], valuesAcc)
        ax[0, 0].set_title('no log')

        plot_acc(ax[0, 1], valuesAcc)
        ax[0, 1].set_title('x log')
        ax[0, 1].set_xscale('symlog', linthresh=1e-4)

        plot_acc(ax[1, 0], valuesAcc)
        ax[1, 0].set_title('y log')
        ax[1, 0].set_yscale('log')

        plot_acc(ax[1, 1], valuesAcc)
        ax[1, 1].set_title('x and y log')
        ax[1, 1].set_xscale('symlog', linthresh=1e-4)
        ax[1, 1].set_yscale('log')

        fig.set_size_inches(15.5, 8.5)
        plt.subplots_adjust(hspace=0.3)
        plt.legend(loc='lower right', bbox_to_anchor=(1.7, 1.65),
                   fontsize="16")  # TODO: change location of legend
        plt.suptitle(
            f'Accuracy vs. Learning Rate for Step Number {step} in Epoch {epoch} of {max_epoch}', fontsize=20)
        fig.savefig(output_path + f'Acc{step}.pdf',
                    dpi=100, bbox_inches='tight')

    # Plot loss
        figloss, axloss = plt.subplots(2, 2)

        valuesLoss = {
            LOSS_LABELS[key]: (LEARNING_RATES, losses[key][i]) for key in LOSS_LABELS
        }

        plot_loss(axloss[0, 0], valuesLoss)
        axloss[0, 0].set_title('no log')

        plot_loss(axloss[0, 1], valuesLoss)
        axloss[0, 1].set_title('x log')
        axloss[0, 1].set_xscale('symlog', linthresh=1e-4)

        plot_loss(axloss[1, 0], valuesLoss)
        axloss[1, 0].set_title('y log')
        axloss[1, 0].set_yscale('symlog', linthresh=1e-4)

        plot_loss(axloss[1, 1], valuesLoss)
        axloss[1, 1].set_title('x and y log')
        axloss[1, 1].set_xscale('symlog', linthresh=1e-4)
        axloss[1, 1].set_yscale('symlog', linthresh=1e-4)

        plt.legend(loc='upper right', bbox_to_anchor=(1.7, 2.35),
                   fontsize=16)  # TODO: change location of legend
        plt.suptitle(
            f'Loss vs. Learning Rate for Step Number {step} in Epoch {epoch} of {max_epoch}', fontsize=20)
        figloss.set_size_inches(15.5, 8.5)
        plt.subplots_adjust(hspace=0.3)
        # figloss.tight_layout()
        figloss.savefig(
            output_path + f'Loss{step}.pdf', dpi=100, bbox_inches='tight')


def initialize_epoch_arrays(max_epoch):
    accuracies = np.zeros(max_epoch)
    losses = np.zeros(max_epoch)
    return accuracies, losses

def initialize_3_epoch_arrays(max_epoch):
    accuracies = np.zeros(max_epoch)
    losses = np.zeros(max_epoch)
    lr=np.zeros(max_epoch)
    return accuracies, losses, lr

# def plot_mine(axis, values):
 #   axis.plot(epoch, accuracies, label="Validation Accuracy")


def plot_training_progress(d2, output_path):
    epoch = np.unique([item['epoch'] for item in d2])
    accuracies, losses = initialize_epoch_arrays(len(epoch))
    for i, item in enumerate(d2):
        accuracies[i] = item['val_acc']
        losses[i] = item['train_loss']

    figacc, axacc = plt.subplots()
    axacc.plot(epoch, accuracies, label="Validation Accuracy")
    axacc.set_title('Validation Accuracy vs. Epoch')
    axacc.set_xlabel('Epoch')
    axacc.set_ylabel('Accuracy')
    axacc.legend()
    figacc.savefig(output_path + 'valAccuracy.pdf',
                   dpi=100, bbox_inches='tight')
    figloss, axloss = plt.subplots()
    axloss.plot(epoch, losses, label="Training Loss")
    axloss.set_title('Training Loss vs. Epoch')
    axloss.set_xlabel('Epoch')
    axloss.set_ylabel('Loss')
    axloss.legend()
    figloss.savefig(output_path + 'trainLoss.pdf',
                    dpi=100, bbox_inches='tight')


def plot_comparison(d_armijo, d_loss_sls, d_acc_sls, output_path):
    #with plt.rc_context(bundles.beamer_moml()):
    #with plt.rc_context(bundles.iclr2023()):
    #with plt.rc_context(bundles.neurips2021(usetex=True, family="serif")):
    epoch = np.unique([item['epoch'] for item in d_armijo])
    accuracies_armijo, losses_armijo, lr_armijo = initialize_3_epoch_arrays(
        len(epoch))
    accuracies_loss_sls, losses_loss_sls, lr_loss_sls = initialize_3_epoch_arrays(
        len(epoch))
    accuracies_acc_sls, losses_acc_sls, lr_acc_sls = initialize_3_epoch_arrays(
        len(epoch))
    for i, item in enumerate(d_armijo):
        accuracies_armijo[i] = item['val_acc']
        losses_armijo[i] = item['train_loss']
        lr_armijo[i] = item['step_size']
    for i, item in enumerate(d_loss_sls):
        accuracies_loss_sls[i] = item['val_acc']
        losses_loss_sls[i] = item['train_loss']
        lr_loss_sls[i] = item['step_size']
    for i, item in enumerate(d_acc_sls):
        accuracies_acc_sls[i] = item['val_acc']
        losses_acc_sls[i] = item['train_loss']
        lr_acc_sls[i] = item['step_size']
    # for validation accuracy
    figacc, axacc = plt.subplots()
    #for different optimizers
    #axacc.plot(epoch, accuracies_armijo, label="SGD Armijo")
    #axacc.plot(epoch, accuracies_loss_sls, label="Loss based SLS")
    #axacc.plot(epoch, accuracies_acc_sls, label="Weak Accuracy based SLS")

    #change batches
    axacc.plot(epoch, accuracies_armijo, label="on same Batch")
    axacc.plot(epoch, accuracies_loss_sls, label="on different (same sized) Batch")
    axacc.plot(epoch, accuracies_acc_sls, label="on big Batch")
    
    axacc.set_xlabel('Epoch')
    axacc.set_ylabel('Validation Accuracy')
    axacc.legend()
    #axacc.grid()
    figacc.savefig(output_path + 'valAccuracy.pdf',
                dpi=100, bbox_inches='tight')
    # for training loss
    figloss, axloss = plt.subplots()
    #axloss.plot(epoch, losses_armijo, label="SGD Armijo")
    #axloss.plot(epoch, losses_loss_sls, label="Loss based SLS")
    #axloss.plot(epoch, losses_acc_sls, label="Weak Accuracy based SLS")
    axloss.plot(epoch, losses_armijo, label="on same Batch")
    axloss.plot(epoch, losses_loss_sls, label="on different (same sized) Batch")
    axloss.plot(epoch, losses_acc_sls, label="on big Batch")
    axloss.set_title('Training Loss vs. Epoch')
    axloss.set_xlabel('Epoch')
    axloss.set_ylabel('Training Loss')
    axloss.legend()
    #axloss.grid()
    figloss.savefig(output_path + 'trainLoss.pdf',
                    dpi=100, bbox_inches='tight')
    # for learning rate
    figlr, axlr = plt.subplots()
    #axlr.plot(epoch, lr_armijo, label="SGD Armijo")
    #axlr.plot(epoch, lr_loss_sls, label="Loss based SLS")
    #axlr.plot(epoch, lr_acc_sls, label="Weak Accuracy based SLS")
    axlr.plot(epoch, lr_armijo, label="on same Batch")
    axlr.plot(epoch, lr_loss_sls, label="on different (same sized) Batch")
    axlr.plot(epoch, lr_acc_sls, label="on big Batch")
    axlr.set_title('Learning Rate vs. Epoch')
    axlr.set_xlabel('Epoch')
    axlr.set_ylabel('Learning Rate')
    axlr.set_yscale('log')
    axlr.legend()
    #axlr.grid()
    figlr.savefig(output_path + 'learningRate.pdf',
                dpi=100, bbox_inches='tight')

def slope(d,slope):
    epoch = np.unique([item['epoch'] for item in d])
    #print(epoch)
    acc=[None]*len(epoch)
    loss=[None]*len(epoch)
    #for i in slopes:
     #   acc[i]=[]
     #   loss[i]=[]
    for j, item in enumerate(d):
        acc[j] = item['val_acc']
        loss[j] = item['train_loss']
    print(f'maximal accuracy for slope {slope} is {max(acc)}')
    print(f'minimal loss for slope {slope} is {min(loss)}')
    return loss, acc
    #TODO slopes noch plotten!

def print_slope(output_path, loss_list, acc_list, slopes):
    epochs=range(len(loss_list[0]))
    max_acc=[None]*len(slopes)
    min_loss=[None]*len(slopes)
    figacc, axacc = plt.subplots()
    figloss, axloss = plt.subplots()
    axacc.set_title('Validation Accuracy vs. Epoch for different Slopes')
    axloss.set_title('Training Loss vs. Slope')
    for i in range(len(slopes)):
        axacc.plot(epochs[1:], acc_list[i][1:], label=f"Slope: {slopes[i]}")          
        axloss.plot(epochs[1:], loss_list[i][1:], label=f"Slope: {slopes[i]}")
        max_acc[i]=max(acc_list[i])
        min_loss[i]=min(loss_list[i])
        #plot the accuracy and loss over the epochs for different slopes in seperate plots
        figacc1, axacc1 = plt.subplots()
        figloss1, axloss1 = plt.subplots()
        axacc1.plot(epochs, acc_list[i], label=f"Slope: {slopes[i]}")
        axloss1.plot(epochs, loss_list[i], label=f"Slope: {slopes[i]}")
        axacc1.set_title(f'Validation Accuracy vs. Epoch for slope {slopes[i]}')
        axloss1.set_title(f'Training Loss vs. Epoch for slope {slopes[i]}')
        axacc1.set_xlabel('Epochs')
        axacc1.set_ylabel('Accuracy')
        axacc1.legend()
        axloss1.set_xlabel('Epochs')
        axloss1.set_ylabel('Loss')
        axloss1.legend()
        figacc1.savefig(output_path + f'AccuracySlope{slopes[i]}.pdf',
                    dpi=100, bbox_inches='tight')
        figloss1.savefig(output_path + f'LossSlope{slopes[i]}.pdf',
                    dpi=100, bbox_inches='tight')       
        
    axacc.legend()
    axacc.set_xlabel('Epochs')
    axacc.set_ylabel('Accuracy')
    figacc.savefig(output_path + 'AccuracySlopesCutted.pdf',
                    dpi=100, bbox_inches='tight')
    axloss.set_xlabel('Epochs')
    axloss.set_ylabel('Loss')
    axloss.legend()
    figloss.savefig(output_path + 'LossSlopesCutted.pdf',
                    dpi=100, bbox_inches='tight')
    figmax, axmax = plt.subplots()
    figmin, axmin = plt.subplots()
    axmax.plot(slopes, max_acc, '-o', label="Maximal Accuracy")
    axmin.plot(slopes, min_loss, '-o', label="Minimal Loss")
    axmax.set_xlabel('Slope')
    axmax.set_ylabel('Accuracy')
    axmin.set_xlabel('Slope')
    axmin.set_ylabel('Loss')
    axmax.legend()
    axmin.legend()
    figmax.savefig(output_path + 'maxAccuracy.pdf',
                    dpi=100, bbox_inches='tight')
    figmin.savefig(output_path + 'minLoss.pdf',
                    dpi=100, bbox_inches='tight')
    print(f'max accuracy for different slopes:{max_acc}')
    #print min loss for different slopes with only 2 decimals
    print(f'min loss for different slopes:{[round(num, 5) for num in min_loss]}')
    
    
def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    # saves the optimal learning rate
    optimal_lr.append(xmax)
    ymax = y.max()
    # text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    improve = ymax-y[6]
    improvement.append(improve)
    text = f"highest accuracy ({ymax}) for lr: {xmax} \n difference between max and 0: {improve}"
    # text= "difference between max and 0: {f}".format(f=xmax-x[6])
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    arrowprops = dict(
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.16), **kw)


def plot_optimal_lr(output_path):
    fig, ax = plt.subplots()
    epochs_lr_set = set(epochs_lr)
    epochs_use = list(epochs_lr_set)
    epochs_use.sort()
    optimal_lr_set = optimal_lr[0::4]
    # print(optimal_lr_set)
    # print(epochs_use)
    ax.plot(epochs_use, optimal_lr_set, '-o')
    ax.set_xlabel('epoch')
    ax.set_ylabel('optimal learning rate')
    ax.set_title('lr wich gives the highest accuracy')
    fig.savefig(output_path + 'OptimalLr.pdf', dpi=100, bbox_inches='tight')


def plot_improvement(output_path):
    print('plotting improvement...')
    plt.rcParams.update({"figure.dpi": 150})
    #plt.rcParams.update(figsizes.cvpr2022_full())
    plt.rcParams.update(fontsizes.neurips2021())
    plt.rcParams.update(fonts.neurips2021())
    plt.rcParams.update(figsizes.neurips2021())
    fig, ax = plt.subplots()
    epochs_lr_set = set(epochs_lr)
    epochs_use = list(epochs_lr_set)
    epochs_use.sort()
    improvement_set = improvement[0::4]
    # print(improvement_set)
    ax.plot(epochs_use, improvement_set, '-o')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Improvement')
    ax.set_title(
        'Improvement of the accuracy compared to the accuracy with lr=0')
    fig.savefig(output_path + 'Improvement.pdf', dpi=100, bbox_inches='tight')


def plot_optimal_lr_improvement(output_path, exp):
    print('plotting optimal lr and improvement...')
    plt.rcParams.update({"figure.dpi": 150})
    #plt.rcParams.update(figsizes.cvpr2022_full())
    plt.rcParams.update(fontsizes.neurips2021())
    plt.rcParams.update(fonts.neurips2021())
    plt.rcParams.update(figsizes.neurips2021())
    plt.rcParams.update(figsizes.neurips2021())
    fig, ax = plt.subplots()
    epochs_lr_set = set(epochs_lr)
    epochs_use = list(epochs_lr_set)
    epochs_use.sort()
    optimal_lr_ = optimal_lr[0::4]
    improvement_ = improvement[0::4]

    print(exp)
    print(epochs_use)
    print(optimal_lr_)
    print(improvement_)
    ax.plot(epochs_use, optimal_lr_, '-o', label='optimal learning rate')
    ax.plot(epochs_use, improvement_, '-o', label='improvement')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Improvement')
    ax.set_title(
        'Improvement of the accuracy compared to the accuracy with lr=0')
    ax.legend()
    fig.savefig(output_path + 'OptimalLrImprovement.pdf',
                dpi=100, bbox_inches='tight')


def main():
    parser = ArgumentParser()
    parser.add_argument('--exp', type=str, help='experiment to plot')
    parser.add_argument('--compare', type=str,
                        help='compare experiments')
    parser.add_argument('--slope', type=str,
                        help='compare experiments')
    #parser.add_argument('--slope2', type=str,
     #                   help='compare experiments')
    parser.add_argument('--all', action='store_true',
                        help='plot all experiments')
    args = parser.parse_args()

    if args.all:
        print('Plotting all experiments...')
        for exp in EXPERIMENTS:
            os.makedirs(FIGURE_PATH + f'{exp}\\', exist_ok=True)
            file = RESULT_PATH + f'{exp}\\loss_list.pkl'
            file2 = RESULT_PATH + f'{exp}\\score_list.pkl'
            d = pd.read_pickle(file)
            d2 = pd.read_pickle(file2)
            plot_accuracy_and_loss(d, FIGURE_PATH + f'{exp}\\')
            plot_training_progress(d2, FIGURE_PATH + f'{exp}\\')
            plot_middle(RESULT_PATH + f'{exp}\\',
                        FIGURE_PATH + f'{exp}\\middle\\')
            plot_optimal_lr(FIGURE_PATH + f'{exp}\\')
            plot_improvement(FIGURE_PATH + f'{exp}\\')
            plot_optimal_lr_improvement(FIGURE_PATH + f'{exp}\\', exp)

            # empty the lists for the next experiment
            epochs_lr.clear()
            optimal_lr.clear()
            improvement.clear()

    elif args.compare is not None:
        print(f'Comparing different optimizers for {args.compare}...')
        a=args.compare
        #os.makedirs(FIGURE_PATH +'comparison/' f'weak_{args.compare}/', exist_ok=True)
        #os.makedirs(FIGURE_PATH + 'mnist', exist_ok=True)
        os.makedirs(FIGURE_PATH +'changeBatch/' f'Loss_{args.compare}/', exist_ok=True)
        #for plotting the different optimizers
        file_armijo = RESULT_PATH + \
            f'comparison/{a}/armijo/score_list.pkl' #{args.exp}
        file_loss_sls = RESULT_PATH + \
            f'comparison/{a}/loss_sls/score_list.pkl'
        file_acc_sls = RESULT_PATH + \
            f'comparison/{a}/acc_sls/score_list.pkl'
        file_weak_acc_sls = RESULT_PATH + \
            f'comparison/{a}/weak_acc_sls/score_list.pkl'
        d_armijo = pd.read_pickle(file_armijo)
        d_loss_sls = pd.read_pickle(file_loss_sls)
        #d_acc_sls = pd.read_pickle(file_acc_sls)
        d_weak_acc_sls = pd.read_pickle(file_weak_acc_sls)
        
        #for plotting the different batches
        f_same= RESULT_PATH+f'changeBatchLoss/{a}/same/score_list.pkl'
        f_small= RESULT_PATH+f'changeBatchLoss/{a}/small/score_list.pkl'
        f_big= RESULT_PATH+f'changeBatchLoss/{a}/big/score_list.pkl'

        d_same=pd.read_pickle(f_same)
        d_small=pd.read_pickle(f_small)
        d_big=pd.read_pickle(f_big)
        plot_comparison(d_same, d_small, d_big, FIGURE_PATH+f'changeBatch/Loss_{a}/')
        #plot_comparison(d_armijo, d_loss_sls, d_weak_acc_sls,#d_acc_sls,
         #               FIGURE_PATH +'comparison/' f'weak_{a}/')

    elif args.slope is not None:
        print(f'printing slope result for {args.slope}...')
        a=args.slope
        slopes=[0.2, 0.5, 0.7, 0.8, 0.9,1,2,3,4,5,6]
        os.makedirs(FIGURE_PATH +'slope/' f'slope_{a}/', exist_ok=True)
        loss_list=[None]*11
        acc_list=[None]*11
        data_frames=[None]*11
        for i in range(len(slopes)):
            path=RESULT_PATH+f'slope_search/{slopes[i]}/{a}/score_list.pkl'
            data_frames[i]=pd.read_pickle(path)
            loss, acc=slope(data_frames[i], slopes[i])
            loss_list[i]=loss
            acc_list[i]=acc

        print_slope(FIGURE_PATH + f'slope/slope_{a}/', loss_list, acc_list, slopes)

    elif args.exp is not None:
        print(f'Plotting experiment {args.exp}...')
        os.makedirs(FIGURE_PATH + f'{args.exp}/', exist_ok=True)
        file2 = RESULT_PATH + f'{args.exp}/score_list.pkl'
        d2 = pd.read_pickle(file2)
        plot_training_progress(d2, FIGURE_PATH + f'{args.exp}/')
    else:
        parser.error(
            'Please specify an experiment to plot or use --all to plot all experiments.')


if __name__ == '__main__':
    main()


    '''elif args.slope is not None:
        print(f'printing slope result for {args.slope}...')
        a=args.slope
        slopes=[0,1,2,3,4,5]
        os.makedirs(FIGURE_PATH +'slope/' f'slope_{args.compare}/', exist_ok=True)
        data_frames=[None]*6
        for i in slopes:
            path=RESULT_PATH+f'slope_search/{i+1}/{a}/score_list.pkl'
            data_frames[i]=pd.read_pickle(path)
            slope(data_frames[i], i+1)'''