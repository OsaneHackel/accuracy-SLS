import os
import argparse
import torchvision
import pandas as pd
import torch
import numpy as np
import time
import pprint
import tqdm
import exp_configs
import csv
import derivativeTest
from src import datasets, models, optimizers, metrics

import torch.optim.lr_scheduler as lr_scheduler

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj

'''with DerivTest=True, you get the test for the derivative analysis (on the losss function)
    with runtime=True, you get the runtime analysis (as csv file) (only works for my own optimizers)
    with epsilon= ... you can change the epsilon for the derivative analysis'''


def trainval(exp_dict, savedir_base, datadir, reset=False, metrics_flag=True,
             DerivTest=False, epsilon=1e-4, runtime=False,  normalization=True, use_tqdm=True):
    
    slope = exp_dict.get('slope', 1.0)
    version = exp_dict["opt"].get('version', 'accuracy')
    searchBatch = exp_dict.get("searchBatch", 'same')
    print(f"searchBatch: {searchBatch}")

    ########################### configurate the experiment############################################

    # DerivTest = False
    # epsilon = 1e-4

    # runtime=False

    #################################################################################################
    number_of_steps = 0
    test = DerivTest
    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    epoch_times = []
    epoch_grad_normes=[]
    if runtime:
        step_times_means = []
        step_times = []
        step_times_means = []
        number_of_iterations = []
        step_sizes = []
        step_size_diffs = []
        rel_step_size_changes = []
        grad_norms = []
        losses = []

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    informations=dict(exp_dict)
    informations['version'] = version
    informations['searchBatch'] = searchBatch
    informations['normalization'] =normalization
    informations['slope for weak accurracy']=slope
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), informations)
    pprint.pprint(informations)
    print('Experiment saved in %s' % savedir)
    # print('why???')
    # set seed
    # ---------------
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # -----------

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=4,
                                               batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=256,
                                             num_workers=4,
                                             persistent_workers=True)

    # print('datasetLoadingWorked')
    # Model
    # -----------
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).cuda()
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    # print("loadOptimizer")
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch=n_batches_per_epoch)
    
    if exp_dict.get('cosine_annealing', False):
        cosanneal = lr_scheduler.CosineAnnealingLR(opt, T_max=exp_dict['max_epoch'])
        rampup = lr_scheduler.LinearLR(opt, start_factor=1/4, total_iters=7)
        scheduler = lr_scheduler.ChainedScheduler([rampup, cosanneal])
    elif exp_dict["opt"].get('weight_decay', False):
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=exp_dict['lr_milestones'], gamma=0.1)
    else:
        scheduler = None

    # print('optimizer mySGD is loaded')
    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    opt_path = os.path.join(savedir, 'opt_state_dict.pth')

    # daran liegt es, dass die Experimente immer weitergefuert werden
    if os.path.exists(score_list_path):
        # resume experiment
        score_list = hu.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print('Starting experiment at epoch %d/%d' %
          (s_epoch, exp_dict['max_epoch']))
    start=time.time()

    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)

        score_dict = {"epoch": epoch}

        if metrics_flag:
            # 1. Compute train loss over train set
            score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, train_loader,
                                                                         metric_name=exp_dict["loss_func"])

            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_loader,
                                                                      metric_name=exp_dict["acc_func"])

        # 3. Train over train loader
        model.train()
        print("%d - Training model with %s..." %
              (epoch, exp_dict["loss_func"]))

        s_time = time.time()
        results=[]
        # tqdm wird verwendet um die wachsenden weissen Balken zu erzeugen
        #count =0 # diag

        ds = tqdm.tqdm(train_loader) if use_tqdm else train_loader
        for images, labels in ds:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            opt.zero_grad()
            #print(f"optimizer step Nr. {count}") # diag
            # !!!!!!!!!!!!!!!!wichtig hier den neuen Optimierer reinschreiben!
            if exp_dict["opt"]["name"] in exp_configs.ours_opt_list:
                def closure(): return loss_function(model, images, labels, backwards=False)
                start_time = time.time()
                if runtime:
                    if exp_dict["opt"]["name"] in ["Sls"]:
                        loss, iterations_of_step, step_size_log, step_size_old, grad_norm, loss = opt.step(closure, images, labels, model, searchBatch, train_set,
                                                                                                            number_of_steps, normalization, version, slope)
                    else:
                        loss, iterations_of_step, step_size_log, step_size_old = opt.step(
                            closure, images, labels, model)
                elif exp_dict["opt"]["name"] in ["accSls"]+["lossSls"]+["Sls"]:
                    if exp_dict["opt"]["name"] in ["Sls"]:
                        results=opt.step(closure, images, labels, model, searchBatch,
                                 train_set, number_of_steps, normalization, version, slope)
                        epoch_grad_normes.append(results[4].item())
                    else:
                        opt.step(closure, images, labels, model)
                elif exp_dict["opt"]["name"] in ["sgd_armijo"]:
                    results=opt.step(closure)
                    epoch_grad_normes.append(results[1].item())
                else:
                    opt.step(closure)
                    epoch_grad_normes.append(0)
                
                number_of_steps += 1
                end_time = time.time()
                #print(epoch_grad_normes)
                #count+=1 # diag
                #if count ==5: # diag
                #    break    # diag
                # Logs for runtime analysis
                if runtime:
                    runtime_analysis(step_size_log, step_size_old, step_size_diffs, step_sizes, rel_step_size_changes,
                                     step_times, number_of_iterations, iterations_of_step, start_time, end_time, grad_norm, grad_norms, losses, loss)

                # logs for slope analysis
                if test:
                    derivativeTest.test_if_all_equal(
                        model, images, labels, epsilon)
                    test = False

            else:
                loss = loss_function(model, images, labels)
                loss.backward()
                opt.step()

            

        if scheduler is not None:
            scheduler.step()

        e_time = time.time()
        epoch_times.append(e_time-s_time)

        # Record metrics Hier werden die Ergebnisse in das Dictionary geschrieben
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["n_forwards"] = opt.state["n_forwards"]
        score_dict["n_backwards"] = opt.state["n_backwards"]
        score_dict["batch_size"] = train_loader.batch_size
        score_dict["train_epoch_time"] = e_time - s_time
        #score_dict["grad_norm"]=epoch_grad_normes
        score_dict["mean_grad_norm"]= np.mean(epoch_grad_normes)
        epoch_grad_normes=[]

        score_list += [score_dict]  # hier dann in die Datei uebertragen

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())

        # save runtime analysis as csv
        # round the values in step_times
        if runtime:
            save_runtime_analysis(savedir, epoch, step_times, step_times_means, number_of_iterations,
                                  step_sizes, step_size_diffs, rel_step_size_changes, grad_norms, losses)

        test = DerivTest

        print("Saved: %s" % savedir)

        

    if runtime:
        save_epoch_times(epoch_times, step_times_means)
    end=time.time()
    print('Experiment completed')
    informations2=dict(informations)
    informations2['runtime in s'] = end-start
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), informations2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                 savedir_base=args.savedir_base,
                 datadir=args.datadir,
                 reset=args.reset)


def runtime_analysis(step_size_log, step_size_old, step_size_diffs, step_sizes, rel_step_size_changes,
                     step_times, number_of_iterations, iterations_of_step, start_time, end_time, grad_norm, grad_norms, losses, loss):
    step_times.append(end_time-start_time)
    number_of_iterations.append(iterations_of_step)
    step_size_diff = step_size_log-step_size_old
    step_size_diffs.append(step_size_diff)
    step_sizes.append(step_size_log)
    rel_step_size_changes.append(step_size_diff/step_size_old)
    grad_norms.append(grad_norm.item())
    losses.append(loss.item())


def save_runtime_analysis(savedir, epoch, step_times, step_times_means, number_of_iterations, step_sizes, step_size_diffs, rel_step_size_changes, grad_norms, losses):

    # round the values in step_times
    step_times = [round(x*1000, 3) for x in step_times]
    step_sizes = [round(x, 6) for x in step_sizes]
    step_size_diffs = [round(x, 6) for x in step_size_diffs]
    rel_step_size_changes = [round(x, 6) for x in rel_step_size_changes]
    grad_norms = [round(x, 6) for x in grad_norms]
    losses = [round(x, 6) for x in losses]

    # create dataframe
    df = pd.DataFrame({f"step_times_epoch_{epoch}_in_ms": step_times,
                       f"number_of_iterations_epoch_{epoch}": number_of_iterations,
                       f"step_sizes_epoch_{epoch}": step_sizes,
                       f"step_size_diffs_epoch_{epoch}": step_size_diffs,
                       f"rel_step_size_changes_epoch_{epoch}": rel_step_size_changes,
                       f"grad_norm_epoch_{epoch}": grad_norms,
                       f"loss_epoch_{epoch}": losses})

    # save dataframe and empty the lists
    df.to_csv(os.path.join(savedir, f"runtime_analysis{epoch}.csv"))
    # empty step_times
    step_times = []
    number_of_iterations = []
    step_sizes = []
    step_size_diffs = []
    rel_step_size_changes = []
    grad_norms = []
    losses = []

    # write the mean of the step_times in a list
    step_times_means.append(np.mean(step_times))


def save_epoch_times(savedir, epoch_times, step_times_means):
    epoch_times = [round(x, 3) for x in epoch_times]
    step_times_means = [round(x, 3) for x in step_times_means]
    df = pd.DataFrame({"epoch_times_in_s": epoch_times,
                      "step_times_means_in_ms": step_times_means})
    df.to_csv(os.path.join(savedir, "epoch_times.csv"))
