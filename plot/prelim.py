import common 
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from tueplots import figsizes, fontsizes, fonts
plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())

def loss_based_vs_armijo(result_dir, out_dir, dataset, model):  
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    print(f'Printing Loss Based Sls vs. Armijo Sgd on {dataset} with {model}')
    epoch = range(0,200)
    dfs_loss = common.give_score_lists(result_dir,dataset, model, optimizer= "Sls", version = "loss", batch = "same")
    dfs_armijo = common.give_score_lists(result_dir,dataset, model, optimizer="sgd_armijo")
    l_acc=common.give_values(dfs_loss[0], 'val_acc')
    l_loss=common.give_values(dfs_loss[0], 'train_loss')
    l_lr =common.give_values(dfs_loss[0], 'step_size')
    a_acc=common.give_values(dfs_armijo[0], 'val_acc')
    a_loss =common.give_values(dfs_armijo[0], 'train_loss')
    a_lr =common.give_values(dfs_armijo[0], 'step_size')
    fig, ax = plt.subplots(1,2)
    for i,value in enumerate(['Learning Rate','Validation Accuracy']):
        l=[]
        a=[]
        if value == 'Validation Accuracy':
            l=l_acc
            a=a_acc
        elif value == 'Training Loss':
            l=l_loss
            a=a_loss
        else:
            l=l_lr
            a=a_lr
        ax[i].plot(epoch, l, label="Loss Based SLS", linewidth=0.8)
        ax[i].plot(epoch, a, label="SGD Armijo", linewidth=0.8) 
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(value)
        ax[i].legend()
        #ax.set_title(f'Loss Based Sls vs. Armijo Sgd on {dataset} with {model}')
    fig.suptitle(f'Loss Based SLS vs. SGD Armijo on Cifar100')
    fig.savefig(out_dir / f'LBSvsSA.pdf',
                    dpi=100, bbox_inches='tight')
        
def accuracy_based_fail(result_dir, out_dir):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    epoch = range(0,200)
    dfs_mnist=common.give_score_lists(result_dir,'mnist', 'mlp', 'Sls', version='accuracy')
    #dfs_cifar10=common.give_score_lists(result_dir,'cifar10','resnet34', 'Sls', version='accuracy')
    dfs_cifar10_dense=common.give_score_lists(result_dir,'cifar10', 'densenet121', 'Sls', version='accuracy')
    dfs_cifar100=common.give_score_lists(result_dir,'cifar100', 'resnet34_100', 'Sls', version='accuracy')
    lr_mnist=common.give_values(dfs_mnist[0], 'step_size')
    #lr_cifar10=common.give_values(dfs_cifar10[0], 'step_size')
    lr_cifar10_dense=common.give_values(dfs_cifar10_dense[0], 'step_size')
    lr_cifar100=common.give_values(dfs_cifar100[0], 'step_size')
    acc_mnist=common.give_values(dfs_mnist[0], 'val_acc')
    #acc_cifar10=common.give_values(dfs_cifar10[0], 'val_acc')
    acc_cifar10_dense=common.give_values(dfs_cifar10_dense[0], 'val_acc')
    acc_cifar100=common.give_values(dfs_cifar100[0], 'val_acc')
    fig, ax = plt.subplots(1,2)
    for i,label in enumerate(['Learning Rate', 'Validation Accuracy']):
        if label == 'Learning Rate':
            value='lr'
            mnist=lr_mnist
            #cifar10=lr_cifar10
            cifar10_dense=lr_cifar10_dense
            cifar100=lr_cifar100
        else:
            value='acc'
            mnist=acc_mnist
            #cifar10=acc_cifar10
            cifar10_dense=acc_cifar10_dense
            cifar100=acc_cifar100
        
        ax[i].plot(epoch, mnist, label="Mnist", linewidth=1)
        #ax.plot(epoch, cifar10, label="Cifar10 Resnet34")
        ax[i].plot(epoch, cifar10_dense, label="Cifar10 DenseNet121", linewidth=1)
        ax[i].plot(epoch, cifar100, label="Cifar100 Resnet34", linewidth=1)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(label)
        ax[i].set_yscale('log')
        ax[i].legend()
    fig.suptitle(f'Validation Accuracy and Learning Rate of Accuracy Based SLS on Different Datasets')    
    fig.savefig(out_dir / f'failASls.pdf',
                    dpi=100, bbox_inches='tight')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')
    parser.add_argument('--out-dir', type=Path, default='figures\\final')
    args = parser.parse_args()

    #loss_based_vs_armijo(args.result_dir,args.out_dir, 'cifar100', 'resnet34_100')
    accuracy_based_fail(args.result_dir, args.out_dir)
    pass

if __name__ == '__main__':
    main()