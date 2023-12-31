import numpy as np
import sls
import accSls
from src.optimizers import others

import torch



def get_optimizer(opt, params, n_batches_per_epoch=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers   
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch    
    if opt_name == "sgd_armijo":
        if opt_dict.get("infer_c"):
            c = (1e-3) * np.sqrt(n_batches_per_epoch)
        
        opt = sls.Sls(params,
                    c = opt_dict.get("c") or 0.1,
                    n_batches_per_epoch=n_batches_per_epoch,
                    line_search_fn="armijo")

    elif opt_name == "sgd_goldstein":
        opt = sls.Sls(params, 
                      c=opt_dict.get("c") or 0.1,
                      reset_option=opt_dict.get("reset_option") or 0,
                      n_batches_per_epoch=n_batches_per_epoch,
                      line_search_fn="goldstein")

    elif opt_name == "sgd_nesterov":
        opt = sls.SlsAcc(params, 
                            acceleration_method="nesterov")

    elif opt_name == "sgd_polyak":
        opt = sls.SlsAcc(params, 
                         c=opt_dict.get("c") or 0.1,
                         acceleration_method="polyak")
    
    elif opt_name == "seg":
        opt = sls.SlsEg(params, n_batches_per_epoch=n_batches_per_epoch)


    # ===============================================
    # others
    #dont know if I can simply add this like this
    elif opt_name == "mySGD":
        print("initialisierung von SGD")
        opt = others.MySGD(params)
#########################################################################################################3  

    elif opt_name == "Sls":
        opt = accSls.Sls(params)

############################################################################################################3
    elif opt_name == "adam":
        opt = torch.optim.Adam(params)

    elif opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=4e-3, weight_decay=1e-4)

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(params)

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=1e-3)

    elif opt_name == 'sgd0.1':
        opt = torch.optim.SGD(params, lr=0.1)

    elif opt_name == 'sgd0.1_momentum_weight_decay':
        opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    elif opt_name == 'sgd0.01':
        opt = torch.optim.SGD(params, lr=1e-2) #first plots are with lr=0.1 but original paper uses 1e-3

    #elif opt_name == 'sgd' and scheduler == True:
    #    opt = torch.optim.SGD

    elif opt_name == 'adagrad':
        opt = torch.optim.Adagrad(params)

    elif opt_name == 'rms':
        opt = torch.optim.RMSprop(params)

    elif opt_name == 'adabound':
        opt = others.AdaBound(params)
        print('Running AdaBound..')

    elif opt_name == 'amsbound':
        opt = others.AdaBound(params, amsbound=True)

    elif opt_name == 'coin':
        opt = others.CocobBackprop(params)

    elif opt_name == 'l4':
        params = list(params)
        # base_opt = torch.optim.Adam(params)
        base_opt = torch.optim.SGD(params, lr=0.01, momentum=0.5)
        opt = others.L4(params, base_opt)

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
