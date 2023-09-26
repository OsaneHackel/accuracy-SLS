import matplotlib.pyplot as plt
import land_common as lc
from tueplots import figsizes, fontsizes, fonts

plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())

output_path = 'C:\\Users\\osane\\.vscode\\bachelorarbeit\\figures\\final\\'

Batches = {
    'train1': 'actual batch (bs: 128)',
    'train2': 'first batch (bs: 128)',
    'train3': 'second batch (bs: 128)',
    'big_batch': 'big batch (bs: 1000)',
    'val': 'val acc'
}

#Done
def loss_acc(data, opt, step=1953):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    df_acc, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'train1', 'acc', step)
    df_loss, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'train1', 'loss', step)
    fig, ax = plt.subplots(1,2)
    ax[1].plot(df_acc['lr'], df_acc['values'], '-o', markersize=4)
    ax[1].set_xlabel('Learning Rate')
    ax[1].set_ylabel('Validation Accuracy')
    ax[1].set_title('Validation Accuracy')
    ax[0].plot(df_loss['lr'], df_loss['values'], '-o', markersize=4)
    ax[0].set_xlabel('Learning Rate')
    ax[0].set_ylabel('Training Loss')
    ax[0].set_title('Training Loss')
    #fig.set_size_inches(15.5, 5.5)
    #fig.suptitle(f'{data} with {opt} step {step} epoch {epoch} of {max_epoch}')
    fig.suptitle(f'Cifar10 with Sgd0.1 step {step} epoch {epoch} of {max_epoch}')
    fig.savefig(output_path+'LossVsAcc.pdf', dpi=100, bbox_inches='tight')

#Done   
def gap(data, opt):
    plt.rcParams.update(figsizes.neurips2021(nrows=2, ncols=2))
    steps=[100,3906,9765,15624]
    fig, ax = plt.subplots(2,2)
    for i in range(4):
        df_acc, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'big_batch', 'acc', steps[i])
        df_val, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'val', 'acc', steps[i])
        ax[i//2][i%2].plot(df_acc['lr'], df_acc['values'],'-o', markersize=4)
        ax[i//2][i%2].plot(df_val['lr'], df_val['values'],'-o', markersize=4)
        ax[i//2][i%2].set_xlabel('Learning Rate')
        ax[i//2][i%2].set_ylabel('Accuracy')
        ax[i//2][i%2].set_title(f'Step {steps[i]} (Epoch {epoch} of {max_epoch})')
        #ax[i//2][i%2].legend()
    fig.suptitle(f'Gap Between Big Batch and Validation Batch on Cifar10')
    #fig.suptitle(f'Gap Between Big Batch and Validation Batch on Cifar10 with Sgd0.1')

    #fig.set_size_inches(15.5, 8.5)
    plt.legend(['big batch', 'validation batch'], loc='upper right')
    #plt.subplots_adjust(hspace=0.3)
    fig.savefig(output_path+'gap.pdf', dpi=100, bbox_inches='tight')

#Done
def widening2(data, opt):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=1))
    steps=[100,1953,3906,9765,15624]
    fig, ax = plt.subplots()
    for i in range(len(steps)):
        df_acc, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'train1', 'acc', steps[i])
        ax.plot(df_acc['lr'], df_acc['values'],'-o', label=f'Step {steps[i]} (Epoch {epoch} of {max_epoch})')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    #ax.set_title(f'{data} with {opt}')
    ax.legend()
    fig.suptitle('Cifar10 with Sgd0.1')
    fig.savefig(output_path+'widening.pdf', dpi=100, bbox_inches='tight')

#Done
def batches(data, opt, value='acc'):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=1))
    #plt.rcParams.update(fontsizes.neurips2021())
    #plt.rcParams.update(figsizes.cvpr2022_half())
    fig,ax=plt.subplots()
    for key in Batches:
        df_acc, epoch, max_epoch = lc.give_lr_and_values(data, opt, key, value)
        ax.plot(df_acc['lr'], df_acc['values'],'-o', label=Batches[key], markersize=4)
    ax.set_xlabel('Learning Rate')
    if value == 'acc':
        ax.set_ylabel('Accuracy')
    else:
        ax.set_ylabel('Loss')
    #ax.set_title(f'{data} with {opt} in epoch {epoch} of {max_epoch}')
    ax.legend()
    fig.suptitle(f'Cifar10 with Sgd0.1 in epoch {epoch} of {max_epoch}')
    #fig.set_size_inches(9, 5.5)
    if value == 'acc':
        fig.savefig(output_path+f'AccOnBatches.pdf', dpi=100, bbox_inches='tight')
    else:
        fig.savefig(output_path+f'LossOnBatches.pdf', dpi=100, bbox_inches='tight')
 
def middle_plot(data, opt):
    plt.rcParams.update(figsizes.cvpr2022_half())
    fig,ax=plt.subplots()
    df, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'train1', 'acc')
    #print(df['lr'][3:-3])
    print(df['values'][3:-3])
    ax.plot(df['lr'][4:-3], df['values'][4:-3],'-o', markersize=3)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('symlog', linthresh=1e-4)
    #ax.set_yscale('log')
    fig.suptitle(f'{data} with {opt} in epoch {epoch} of {max_epoch} (middle part)')
    #fig.set_size_inches(9, 5.5)
    fig.savefig(output_path+f'middle.pdf', dpi=100, bbox_inches='tight')

def middle_table(data, opt):
    df, epoch, max_epoch = lc.give_lr_and_values(data, opt, 'train1', 'acc')
    print(df)
    df.to_latex(output_path+f'middle.tex')


def main():
    #loss_acc('cifar10', 'sgd0.1')
    widening2('cifar10', 'sgd0.1') 
    batches('cifar10', 'sgd0.1','acc')
    batches('cifar10', 'sgd0.1','loss')
    gap('cifar10', 'sgd0.1')
    #middle_plot('cifar10', 'sgd0.1')
    #middle_table('cifar10', 'sgd0.1')
    pass

if __name__ == '__main__':
    main()