import argparse
from pathlib import Path
import common 
import matplotlib.pyplot as plt
from tueplots import figsizes, fontsizes, fonts
plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())

DAT_MOD=[['mnist', 'mlp'],
                   ['cifar10','resnet34'],
                   ['cifar10', 'densenet121'],
                    ['cifar100', 'resnet34_100']]

def get_different_batches(result_dir, dataset, model, version):
    slope = None
    print(dataset, model)
    print(version)
    if version == 'weak_accuracy':
        slope = '1.0'
    #TODO: nachschauen, ob slope da angegenben
    same=common.give_score_lists(result_dir, dataset, model, 'Sls',slope, version, 'same', epoch=100)
    small=common.give_score_lists(result_dir, dataset, model, 'Sls',slope, version, 'small', epoch=100)
    big=common.give_score_lists(result_dir, dataset, model, 'Sls',slope, version, 'big', epoch=100)
    return same, small, big

#call this for loss based and weak_accuracy based Sls
def plot_batches_acc(result_dir, dataset, model, version, output_path):
    same_df, small_df, big_df = get_different_batches(result_dir, dataset, model, version)
    same=common.give_values(same_df[0], 'val_acc')
    small=common.give_values(small_df[0], 'val_acc')
    big=common.give_values(big_df[0], 'val_acc')
    epoch = range(0,100)
    fig, ax = plt.subplots()
    ax.plot(epoch, same, label="Same batch")
    ax.plot(epoch, small, label="Small batch")
    ax.plot(epoch, big, label="Big batch")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'Validation Accuracy for {version} based SLS ondifferent Batches on {dataset}_{model}')
    ax.legend()
    fig.savefig(output_path / f'batches_val_acc_{version}_{dataset}_{model}.pdf')

def plot_batches_loss(result_dir, dataset, model, version, output_path):
    same_df, small_df, big_df = get_different_batches(result_dir, dataset, model, version)
    same=common.give_values(same_df[0], 'train_loss')
    small=common.give_values(small_df[0], 'train_loss')
    big=common.give_values(big_df[0], 'train_loss')
    epoch = range(0,100)
    fig, ax = plt.subplots()
    ax.plot(epoch, same, label="Same batch")
    ax.plot(epoch, small, label="Small batch")
    ax.plot(epoch, big, label="Big batch")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'Training Loss for {version} based SLS ondifferent Batches on {dataset}_{model}')
    ax.legend()
    fig.savefig(output_path / f'batches_train_loss_{version}_{dataset}_{model}.pdf')


def plot_batches_lr(result_dir, dataset, model, version, output_path):
    same_df, small_df, big_df = get_different_batches(result_dir, dataset, model, version)
    same=common.give_values(same_df[0], 'step_size')
    small=common.give_values(small_df[0], 'step_size')
    big=common.give_values(big_df[0], 'step_size')
    epoch = range(0,100)
    fig, ax = plt.subplots()
    ax.plot(epoch, same, label="Same batch")
    ax.plot(epoch, small, label="Small batch")
    ax.plot(epoch, big, label="Big batch")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Learning Rate for {version} based SLS ondifferent Batches on {dataset}_{model}')
    ax.legend()
    fig.savefig(output_path / f'batches_lr_{version}_{dataset}_{model}.pdf')

def plot_batches(result_dir, dataset, model, version, output_path):
    plot_batches_acc(result_dir, dataset, model, version, output_path)
    plot_batches_loss(result_dir, dataset, model, version, output_path)
    plot_batches_lr(result_dir, dataset, model, version, output_path)

def double_plot(result_dir, dataset, model, version, output_path):
    same_df, small_df, big_df = get_different_batches(result_dir, dataset, model, version)
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    same_acc=common.give_values(same_df[0], 'val_acc')
    small_acc=common.give_values(small_df[0], 'val_acc')
    big_acc=common.give_values(big_df[0], 'val_acc')
    same_lr=common.give_values(same_df[0], 'step_size')
    small_lr=common.give_values(small_df[0], 'step_size')
    big_lr=common.give_values(big_df[0], 'step_size')
    epoch = range(0,100)
    fig,ax=plt.subplots(1,2)
    ax[1].plot(epoch, same_acc, label="Same batch")
    ax[1].plot(epoch, small_acc, label="Small batch")
    ax[1].plot(epoch, big_acc, label="Big batch")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Validation Accuracy')
    ax[1].legend()
    ax[0].plot(epoch, same_lr, label="Same batch")
    ax[0].plot(epoch, small_lr, label="Small batch")
    ax[0].plot(epoch, big_lr, label="Big batch")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Learning Rate')
    fig.suptitle(f'Learning Rate and Validation Accuracy for l-SLS on Cifar10')
    fig.savefig(output_path / 'batchesloss.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')
    parser.add_argument('--out-dir', type=Path, default='figures\\final')
    args = parser.parse_args()
    double_plot(args.result_dir, 'cifar10', 'resnet34', 'loss', args.out_dir)
   # plot_batches(args.result_dir, 'mnist', 'mlp', 'weak_accuracy', args.out_dir)
    #for dataset, model in DAT_MOD:
    #    print(dataset, model)
    #    plot_batches(args.result_dir, dataset, model, 'loss', args.out_dir)
    #    plot_batches(args.result_dir, dataset, model, 'weak_accuracy', args.out_dir)
    


if __name__ == '__main__':
    main()