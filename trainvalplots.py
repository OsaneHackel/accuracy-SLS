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
import copy
from src import datasets, models, optimizers, metrics

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def trainval(exp_dict, savedir_base, datadir, reset=False, metrics_flag=True):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)
    #print('why???')
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
                              batch_size=exp_dict["batch_size"],
                              num_workers=4)

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)
    
    val_loader = torch.utils.data.DataLoader(val_set,
                              batch_size=256,
                              num_workers=4,
                              persistent_workers=True)


    #print('datasetLoadingWorked')
    # Model
    # -----------
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).cuda()
    
    #compute number of parameters
    modelparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {modelparams}')
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    #print("loadOptimizer")
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch =n_batches_per_epoch)

    #print('optimizer mySGD is loaded')
    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    opt_path = os.path.join(savedir, 'opt_state_dict.pth')
    loss_list_path = os.path.join(savedir,'loss_list.pkl')

    #checking path
    print(loss_list_path)

    #daran liegt es, dass die Experimente immer weitergefuert werden
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
    print('Starting experiment at epoch %d/%d' % (s_epoch, exp_dict['max_epoch']))

    
    loss_list = [] #my code
    n=0 #counts the number of update steps
    overall_steps = exp_dict['max_epoch'] * n_batches_per_epoch # computes the overall number of trainings steps
    benchmark_iteration = overall_steps//10 # after how many optimizer steps the benchmark is done
    print(benchmark_iteration)
    #create datastructure to save the loss and accuracy at the different Set parametersteps
    #loss_list = hu.load_pkl(loss_list_path)  #find out why load_pkl function doesn't work ->because file doesn't exists jet
    #datas={'n':,'loss':, 'acc':}

    ls_model = copy.deepcopy(model)
    ls_model.requires_grad_(False)
    ls_model.train()

    for epoch in range(s_epoch, exp_dict['max_epoch']):
        # Set seed
        np.random.seed(exp_dict['runs']+epoch)
        torch.manual_seed(exp_dict['runs']+epoch)
        torch.cuda.manual_seed_all(exp_dict['runs']+epoch)

        score_dict = {"epoch": epoch}

        # 3. Train over train loader
        model.train()
        print("%d - Training model with %s..." % (epoch, exp_dict["loss_func"]))
        
        s_time = time.time()
        train_loss = 0.0
        for batch_idx, (images,labels) in enumerate(tqdm.tqdm(train_loader)):
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
 
            if exp_dict["opt"]["name"] in exp_configs.ours_opt_list + ["l4"] + ["mySGD"]+["accSls"]:#!!!!!!!!!!!!!!!!wichtig hier den neuen Optimierer reinschreiben!
                closure = lambda : loss_function(model, images, labels, backwards=False)
                loss = opt.step(closure)
            else:
                loss = loss_function(model, images, labels)
                loss.backward()
                opt.step()
            train_loss += loss.item()

#################################################################################################################################################################           
            LR_STEPS = [-1, -5e-1, -1e-1, -1e-2, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1] #TODO learningrates anpassen 1e0
            n+=1
            #print(n%benchmark_iteration)
            if n % benchmark_iteration == 0.0 or n == 100: #always make the same number of benchmarks (n== 100 only testphase)
                print('Evaluating line search')

                images2,labels2 = next(iter(train_loader)) #load next minibatch
                images2, labels2 = images2.cuda(), labels2.cuda()
                images3,labels3 = next(iter(train_loader)) #load next minibatch
                images3, labels3 = images3.cuda(), labels3.cuda()
                big_batch_images, big_batch_labels = draw_batch(train_set, 1024)
                big_batch_images, big_batch_labels = big_batch_images.cuda(), big_batch_labels.cuda()
  
                for lr_idx, lr in enumerate(LR_STEPS): #iterate over the learning rates
                    print(f'Step-size: {lr:.2e}')

                    #model learns with the learning rate and the computed gradient 
                    for param_ls, param_train in zip(ls_model.parameters(), model.parameters()):
                        param_ls.data = param_train - lr * param_train.grad
                        #for p in param['params']:
                            #p.data -= lr * g   #use original data (vectors)
                    #compute the loss and accuracy on the minibatch
                    ls_model.train()
                    with torch.no_grad():
                        train1_loss = metrics.softmax_loss(ls_model, images, labels, backwards=False)
                        train1_acc = metrics.softmax_accuracy(ls_model, images, labels) 
                        train2_loss = metrics.softmax_loss(ls_model, images2, labels2, backwards=False)
                        train2_acc = metrics.softmax_accuracy(ls_model, images2, labels2)
                        train3_loss = metrics.softmax_loss(ls_model, images3, labels3, backwards=False)
                        train3_acc = metrics.softmax_accuracy(ls_model, images3, labels3)
                        big_batch_loss = metrics.softmax_loss(ls_model, big_batch_images, big_batch_labels, backwards=False)
                        big_batch_acc = metrics.softmax_accuracy(ls_model, big_batch_images, big_batch_labels)

                    #compute the loss and accuracy on the train- or validation set     
                    val_loss = metrics.compute_metric_on_dataset(ls_model, val_loader,
                                                metric_name=exp_dict["loss_func"], max_samples=4096)
                    val_acc = metrics.compute_metric_on_dataset(ls_model, val_loader,
                                                      metric_name=exp_dict["acc_func"], max_samples=4096)
                    loss_list.append({
                        'train1_loss': train1_loss.item(),
                        'train1_acc': train1_acc.item(),
                        'train2_loss': train2_loss.item(),
                        'train2_acc': train2_acc.item(),
                        'train3_loss': train3_loss.item(),
                        'train3_acc': train3_acc.item(),
                        'big_batch_loss': big_batch_loss.item(),
                        'big_batch_acc': big_batch_acc.item(),
                        'val_acc': val_acc,
                        'val_loss':val_loss,
                        'number_of_steps': n,
                        'epoch': epoch,
                        'max_epoch': exp_dict["max_epoch"],
                        'batch_size': exp_dict["batch_size"],
                        'lr': lr,
                        'lr_idx': lr_idx,
                    })
                    

                #g=torch.autograd.grad(model.forward(images), images)

#################################################################################################################################################################            

        if metrics_flag:
            score_dict["train_loss"] = train_loss / (batch_idx + 1)
            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_loader,
                                                        metric_name=exp_dict["acc_func"])



        e_time = time.time()

        # Record metrics Hier werden die Ergebnisse in das Dictionary geschrieben
        score_dict["step_size"] = opt.state["step_size"]
        score_dict["n_forwards"] = opt.state["n_forwards"]
        score_dict["n_backwards"] = opt.state["n_backwards"]
        score_dict["batch_size"] =  train_loader.batch_size
        score_dict["train_epoch_time"] = e_time - s_time

        score_list += [score_dict] #hier dann in die Datei uebertragen

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.save_pkl(loss_list_path, loss_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved: %s" % savedir)

    print('Experiment completed')



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


def draw_batch(train_set,n):
    #draws a batch of size n from the training set
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=n, shuffle=True,
        num_workers=4, pin_memory=True)
    images, labels = next(iter(train_loader))
    return images, labels