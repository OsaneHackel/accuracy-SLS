import argparse
from pathlib import Path
import common 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tueplots import figsizes, fontsizes, fonts
plt.rcParams.update({"figure.dpi": 150})
#plt.rcParams.update(figsizes.cvpr2022_full())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(fonts.neurips2021())
Exps={'dat_model':[['mnist', 'mlp'],
                   ['cifar10','resnet34'],
                   ['cifar10', 'densenet121'],
                    ['cifar100', 'resnet34_100']],
      'opt':[{'name':'adamw'}, 
             {'name':'sgd0.1', 'weight_decay':False},
             {'name':'sgd0.1', 'weight_decay':True},
             {"name":"Sls",'slope':'1.0',"version":"weak_accuracy"}, 
             {"name":"Sls","version":"loss"},
             {"name":"sgd_armijo"}]}
Sls={'dat_model':[['mnist', 'mlp'],
                   ['cifar10','resnet34'],
                   ['cifar10', 'densenet121'],
                    ['cifar100', 'resnet34_100']],
        'opt':[{"name":"Sls",'slope':'1.0',"version":"weak_accuracy"},
                {"name":"Sls","version":"loss"},
                {"name":"sgd_armijo"}]}

Opts=['Adam', 'Sgd0.1', 'Sgd0.1 with Wd', 'sa-SLS', 'l-SLS', 'Sgd Armijo']
Sls_Opts=['sa-SLS', 'l-SLS', 'Sgd Armijo']
def max_values_acc(result_dir, dataset, model, optimizer, slope = None, version = None, batch=None, weight_decay=None, epoch=200):
    score_list = common.give_score_lists(result_dir, dataset, model, optimizer, slope, version, batch, weight_decay)
    accs = []
    if len(score_list) != 4:
        print('not all 4 score lists found')

    for dct in score_list:
        acc=common.give_values(dct, 'val_acc')
        accs.append(np.max(acc))
    return accs

def runtime(result_dir, dataset, model, optimizer, slope = None, version = None, batch=None, weight_decay=None, epoch=200):
    exp_dicts=common.give_exp_dict(result_dir, dataset, model, optimizer, slope, version, batch, weight_decay)
    runtimes=[]
    if len(exp_dicts) != 4:
        print('not all 4 score lists found')
    for dct in exp_dicts:
        runtimes.append(dct['runtime in s'])
    return runtimes

def mean_passes(result_dir, dataset, model, optimizer, slope = None, version = None, batch=None, weight_decay=None):
    score_lists = common.give_score_lists(result_dir, dataset, model, optimizer, slope, version, batch, weight_decay)
    forward_passes=[]
    backward_passes=[]
    if len(score_lists) != 4:
        print('not all 4 score lists found')
    for dct in score_lists:
        forward=common.give_values(dct, 'n_forwards')
        backward=common.give_values(dct, 'n_backwards')
        forward_passes.append(forward[-1])
        backward_passes.append(backward[-1])
    mean_fw=np.mean(forward_passes)
    mean_bw=np.mean(backward_passes)
    if dataset=='mnist':
        steps=200*468
        print('mnist')  
    else:
        steps=200*390
    fw_steps=mean_fw/steps
    bw_steps=mean_bw/steps
    return mean_fw, mean_bw, fw_steps, bw_steps

def mean_sd(accs):
    mean_acc = np.mean(accs)
    sd_acc = np.std(accs)
    return mean_acc, sd_acc

def name_exp(dat, model, opt):
    #name = dat + '_' + model
    name=''
    name += opt['name'] if 'name' in opt else ''
    name += '_' + str(opt['slope']) if 'slope' in opt else ''
    name += '_' + str(opt['version']) if 'version' in opt else ''
    name += '_' + str(opt['batch']) if 'batch' in opt else ''
    name += '_' + str(opt['weight_decay']) if 'weight_decay' in opt else ''
    return name


#TODO: raw results for each experiment also in table
def benchmarking(result_dir,Exps, out_dir):
    dat_model= Exps['dat_model']
    opts=Exps['opt']
    dct={}
    for dat, model in dat_model:
        for i,opt in enumerate(opts):
            name=name_exp(dat, model, opt)
            exp=[result_dir, 
                dat,
                model, 
                opt['name'], 
                opt.get('slope'),
                opt.get('version'), 
                opt.get('batch'), 
                opt.get('weight_decay')]
            accs=max_values_acc(*exp)
            mean, sd=mean_sd(accs)
            rt=runtime(*exp)
            mean_rt=np.mean(rt)
            accs.sort()
            print(accs)
            accs_str = [f'{acc*100:.2f}' for acc in accs]
            tmpdct={'Accuracies in %':accs_str, 'Mean Acc':f'{mean*100:.2f}', 'Sd Acc':f'{sd*100:.3f}', 'Mean Runtime in h':f'{mean_rt/3600:.2f}'}
            dct[Opts[i]]=tmpdct
            print(f'Name: {name} with max values {accs}, mean {mean}, sd {sd} and mean_rt {mean_rt/3600}')
        df=pd.DataFrame.from_dict(dct, orient='index')
        dir=f'benchmarking.{dat}_{model}.csv'
        dir2=f'benchmarking.{dat}_{model}.tex'
        df.to_csv(out_dir/dir)
        df.to_latex(out_dir/dir2)
    #df.to_excel(out_dir/'benchmarking.xlsx')
        print(df)            
def lr_crush_mnist(result_dir, output_dir):
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    epochs=range(0,200)
    fig, ax = plt.subplots(1,2)
    for i in [0,1,2,3]:
        score_list= common.give_score_lists(result_dir,
                                            'mnist', 
                                            'mlp',
                                                "Sls",
                                            '1.0',
                                            "weak_accuracy",
                                                'same',
                                            run=i)
        lr=common.give_values(score_list[0], 'step_size')
        accs=common.give_values(score_list[0], 'val_acc')    
        ax[0].plot(epochs, lr,linewidth=0.8)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Learning Rate')
        ax[0].set_yscale('log')
        ax[1].plot(epochs, accs, label=f'Run {i}', linewidth=0.8)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Validation Accuracy')
    ax[1].legend()
    fig.suptitle(f'Learning Rate and Validation Accuracy for sa-SLS on Mnist')
    fig.savefig(output_dir / f'lrCrash.pdf')



def passes(result_dir,output_dir, Sls):
    dat_model= Sls['dat_model']
    opts=Sls['opt']
    dct={}
    for dat, model in dat_model:
        for i,opt in enumerate(opts):
            name=name_exp(dat, model, opt)
            fw,bw,fw_steps, bw_steps=mean_passes(result_dir, 
                dat,
                model, 
                opt['name'], 
                opt.get('slope'),
                opt.get('version'), 
                opt.get('batch'), 
                opt.get('weight_decay'))
            tmpdct={'fw passes':fw,'fw per steps':fw_steps, 'bw passes':bw, 'bw per steps':bw_steps}
            dct[f'{Sls_Opts[i]}']=tmpdct
            print(f'Name: {name} with mean forward {fw} per step: {fw_steps}, mean backward {bw} per step: {bw_steps}')
        df=pd.DataFrame.from_dict(dct, orient='index')
        df.to_csv(output_dir/f'mean_passes_{dat}_{model}.csv')
        df.to_latex(output_dir/f'mean_passes_{dat}_{model}.tex')
    #df.to_excel(output_dir/'mean_passes.xlsx')
    print(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, default='results')
    parser.add_argument('--out-dir', type=Path, default='figures\\final')
    args = parser.parse_args()
    #lr_crush_mnist(args.result_dir, args.out_dir)
    passes(args.result_dir, args.out_dir, Sls)
    #benchmarking(args.result_dir, Exps, args.out_dir)
    pass


if __name__ == '__main__':
    main()