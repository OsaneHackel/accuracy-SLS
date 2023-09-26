import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes, fontsizes, fonts



plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())

output_path = 'C:\\Users\\osane\\.vscode\\bachelorarbeit\\figures\\final\\'
def sigmoid(x, slope=1):
    return 1 / (1 + np.exp(-slope*x))

def plot_sigmoid():
    plt.rcParams.update(figsizes.cvpr2022_half(nrows=1, ncols=1))
    x = np.linspace(-1,1,200)
    fig, ax = plt.subplots()
    ax.plot(x, sigmoid(x, 0.3), label='s = 0.3')
    ax.plot(x, sigmoid(x, 1), label='s = 1')
    ax.plot(x, sigmoid(x, 5), label='s = 5')
    ax.plot(x, sigmoid(x, 10), label='s = 10')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.suptitle(f'Sigmoid Function with Slope Factor s')
    fig.savefig(output_path+'sigmoid.pdf', dpi=100, bbox_inches='tight')

def plot_log_loss():
    plt.rcParams.update(figsizes.cvpr2022_half(nrows=1, ncols=1))
    x = np.linspace(0,1,200)
    fig, ax = plt.subplots()
    ax.plot(x, -np.log(x), label='CE')
    ax.set_xlabel('model output for correct label')
    ax.set_ylabel('loss')
    fig.suptitle(f'Cross Entropy Loss')
    fig.savefig(output_path+'loss.pdf', dpi=100, bbox_inches='tight')

def focal_loss(x, gamma=2):
    return -(1-x)**gamma * np.log(x)

def plot_focal_loss():
    plt.rcParams.update(figsizes.cvpr2022_half(nrows=1, ncols=1))
    x = np.linspace(0,1,200)
    fig, ax = plt.subplots()
    ax.plot(x, -np.log(x), label='CE')
    ax.plot(x, focal_loss(x, 0.5), label='FL, gamma=0.5')
    ax.plot(x, focal_loss(x, 1), label='FL, gamma=1')
    ax.plot(x, focal_loss(x, 2), label='FL, gamma=2')
    ax.plot(x, focal_loss(x, 5), label='FL, gamma=5')
    ax.set_xlabel('model output for correct label')
    ax.set_ylabel('loss')
    ax.legend()
    fig.suptitle(f'Focal Loss')
    fig.savefig(output_path+'focal_loss.pdf', dpi=100, bbox_inches='tight')

def main():
    plot_focal_loss()
    plot_log_loss()
    plot_sigmoid()
    pass

if __name__ == '__main__':
    main()