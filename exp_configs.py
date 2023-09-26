import os
import itertools

from haven import haven_utils as hu

ours_opt_list = ["Sls", "sgd_armijo"]

run_list = [0,1,2,3]
search_slope=[0.1,0.5,0.9,1,2,5,10]
opt_list_sgd=[{"name":"sgd0.1", "weight_decay":False},{"name":"sgd0.1", "weight_decay":True}]#{"name":'sgd0.1_momentum_weight_decay'}
opt_list_sls=[{"name":"Sls","version":"weak_accuracy"}, {"name":"Sls","version":"loss"},{"name":"sgd_armijo"}]
batches=["same", "small", "big"]


EXP_GROUPS = {
        "cifar10_adam": {"dataset":["cifar10"],
                         "model": ["resnet34"],
                        "loss_func":["softmax_loss"],   
                        "opt":[{"name":"adamw"}],
                        "acc_func":["softmax_accuracy"],
                        "cosine_annealing":True,
                        "batch_size":[128],
                        "max_epoch":[200],
                        "runs": run_list},
        "cifar10_sgd": {"dataset":["cifar10"],
                         "model": ["resnet34"],
                        "loss_func":["softmax_loss"],   
                        "opt":opt_list_sgd,
                        "lr_milestones":[[60,120,160]],
                        "acc_func":["softmax_accuracy"],
                        "batch_size":[128],
                        "max_epoch":[200],
                        "runs":run_list},
        "cifar10_sls": {"dataset":["cifar10"],
                        "model": ["resnet34"],
                        "loss_func":["softmax_loss"],
                        "opt":opt_list_sls,
                        "acc_func":["softmax_accuracy"],
                        "batch_size":[128],
                        "max_epoch":[200],
                        "runs":run_list},
        "cifar10_sls_acc": {"dataset":["cifar10"],
                        "model": ["resnet34"],
                        "loss_func":["softmax_loss"],
                        "opt":[{"name":"Sls","version":"accuracy"}],
                        "acc_func":["softmax_accuracy"],
                        "batch_size":[128],
                        "max_epoch":[200],
                        "runs":[0]},
}
#Hauptexperimente
#mnist
EXP_GROUPS['mnist_adam'] = EXP_GROUPS['cifar10_adam'].copy()
EXP_GROUPS['mnist_adam'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})

EXP_GROUPS['mnist_sgd'] = EXP_GROUPS['cifar10_sgd'].copy()
EXP_GROUPS['mnist_sgd'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})

EXP_GROUPS['mnist_sls'] = EXP_GROUPS['cifar10_sls'].copy()
EXP_GROUPS['mnist_sls'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})

EXP_GROUPS['mnist_sls_acc'] = EXP_GROUPS['cifar10_sls_acc'].copy()
EXP_GROUPS['mnist_sls_acc'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})

#Cifar10
#dense net
EXP_GROUPS['cifar10_adam_dense'] = EXP_GROUPS['cifar10_adam'].copy()
EXP_GROUPS['cifar10_adam_dense'].update({
        "model": ["densenet121"]})
EXP_GROUPS['cifar10_sgd_dense'] = EXP_GROUPS['cifar10_sgd'].copy()
EXP_GROUPS['cifar10_sgd_dense'].update({
        "model": ["densenet121"]})
EXP_GROUPS['cifar10_sls_dense'] = EXP_GROUPS['cifar10_sls'].copy()
EXP_GROUPS['cifar10_sls_dense'].update({
        "model": ["densenet121"]})
EXP_GROUPS['cifar10_sls_acc_dense'] = EXP_GROUPS['cifar10_sls_acc'].copy()
EXP_GROUPS['cifar10_sls_acc_dense'].update({
        "model": ["densenet121"]})

#Cifar100
EXP_GROUPS['cifar100_adam'] = EXP_GROUPS['cifar10_adam'].copy()
EXP_GROUPS['cifar100_adam'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})
EXP_GROUPS['cifar100_sgd'] = EXP_GROUPS['cifar10_sgd'].copy()
EXP_GROUPS['cifar100_sgd'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})
EXP_GROUPS['cifar100_sls'] = EXP_GROUPS['cifar10_sls'].copy()
EXP_GROUPS['cifar100_sls'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})
EXP_GROUPS['cifar100_sls_acc'] = EXP_GROUPS['cifar10_sls_acc'].copy()
EXP_GROUPS['cifar100_sls_acc'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})


#Slope search
EXP_GROUPS['cifar10_slope'] = EXP_GROUPS['cifar10_sls'].copy()
EXP_GROUPS['cifar10_slope'].update({
        "opt": [{"name":"Sls","version":"weak_accuracy"}],
        "slope": search_slope,
        "max_epoch": [100],
        "runs": [0]})
EXP_GROUPS['mnist_slope'] = EXP_GROUPS['cifar10_slope'].copy()
EXP_GROUPS['mnist_slope'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})
EXP_GROUPS['cifar100_slope'] = EXP_GROUPS['cifar10_slope'].copy()
EXP_GROUPS['cifar100_slope'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})
EXP_GROUPS['cifar10_slope_dense'] = EXP_GROUPS['cifar10_slope'].copy()
EXP_GROUPS['cifar10_slope_dense'].update({
        "model": ["densenet121"]})

#Batch search
EXP_GROUPS['cifar10_batch'] = EXP_GROUPS['cifar10_sls'].copy()
EXP_GROUPS['cifar10_batch'].update({
        "opt": [{"name":"Sls","version":"weak_accuracy"}, {"name":"Sls","version":"loss"}],
        "max_epoch": [100],
        "searchBatch":batches,
        "runs": [0]})
EXP_GROUPS['mnist_batch'] = EXP_GROUPS['cifar10_batch'].copy()
EXP_GROUPS['mnist_batch'].update({
        "dataset": ["mnist"],
        "model": ["mlp"]})
EXP_GROUPS['cifar100_batch'] = EXP_GROUPS['cifar10_batch'].copy()
EXP_GROUPS['cifar100_batch'].update({
        "dataset": ["cifar100"],
        "model": ["resnet34_100"]})
EXP_GROUPS['cifar10_batch_dense'] = EXP_GROUPS['cifar10_batch'].copy()
EXP_GROUPS['cifar10_batch_dense'].update({
        "model": ["densenet121"]})


EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}


        # MNIST
'''    "cifar10":{"dataset":["cifar10"],
            "model":["resnet34"],
            "loss_func": ["softmax_loss"],
            "opt":["sgd_armijo"],
            "acc_func":["softmax_accuracy"],
            "batch_size":[128],
            "max_epoch":[200],
            "runs":[0,1,6,7]},'''