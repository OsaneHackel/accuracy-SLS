import torch
import copy
import time
#from .metrics import soft_accuracy as sa  # import soft_accuracy

import accSls.utils as ut

#definition of accuracy computation
def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

class AccuracySls(torch.optim.Optimizer):
    """Implements accuracy based stochastic line search.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        beta_f (float, optional): factor used by Goldstein for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        eta_max (float, optional): an upper bound used by Goldstein on the step size (default: 10)
        bound_step_size (bool, optional): a flag used by Goldstein for whether to bound the step-size (default: True)
        line_search_fn (float, optional): the condition used by the line-search to find the 
                    step-size (default: Armijo)
        lr_step_schedule(string, optional): the schedule used to update the step-size (constant, multiplicative, scaling)(default: constant)
                    what works: mnist: constant with gamma 0.003

            mnist:                |          gammas which work
            ----------------------|----------------------------------------
            constant              |    0.003   
            multiplicative        |    !2  !10000 !10e10 !10e15(to large) 9.9995e15 9.995e15 !1.14e14 !10e14  !1000e14 !10000000000e14 !1e15 !5e15 !8e15 !9.5 
            scaling               |    !1.07  !1.1
            --------------------- | ----------------------------------------
            cifar10:              |          gammas which work
            ----------------------|----------------------------------------
            constant              |    0.003
            multiplicative        |    0.003, 0.001, 0.0003, 0.0001
            scaling               |    0.003, 0.001, 0.0003, 0.0001
            --------------------- | ----------------------------------------
            cifar100:             |          gammas which work
            ----------------------|----------------------------------------
            constant              |    0.003
            multiplicative        |    0.003, 0.001, 0.0003, 0.0001
            scaling               |    0.003, 0.001, 0.0003, 0.0001
            --------------------- | ----------------------------------------
    """


    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="armijo",
                 lr_step_schedule='multiplicative'):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        c=c,
                        beta_b=beta_b,
                        gamma=gamma,
                        beta_f=beta_f,
                        reset_option=reset_option,
                        eta_max=eta_max,
                        bound_step_size=bound_step_size,
                        line_search_fn=line_search_fn,
                        lr_step_schedule=lr_step_schedule)
        super().__init__(params, defaults)       

        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

    def step(self, closure, images, labels, model):
        # deterministic closure
        step_size_diff = 0
        number_of_iterations=0
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # loop over parameter groups
        for group in self.param_groups:
            params = group["params"]
            
            # save the current parameters:
            params_current = copy.deepcopy(params)
            grad_current = ut.get_grad_list(params)

            grad_norm = ut.compute_grad_norm(grad_current)
            
            #reset of the stepsize: 0 (no reset), 1 (reset by gamma), 2 (reset to init_step_size)
            #print("init_step_size", group['init_step_size'])
            '''step_size = ut.reset_step(step_size=batch_step_size,
                                    n_batches_per_epoch=group['n_batches_per_epoch'],
                                    gamma=group['gamma'],
                                    reset_option=group['reset_option'],
                                    init_step_size=group['init_step_size'])'''
            
            gamma = group['gamma']
            n_batches_per_epoch = group['n_batches_per_epoch'] #500

            step_size = ut.lr_scheduler(option=group['lr_step_schedule'],batch_step_size=batch_step_size,
                                       n_batches_per_epoch = group['n_batches_per_epoch'],
                                       gamma = group['gamma'])
            #adds a constant to the stepsize
            

            #print(2**(1. / n_batches_per_epoch))
            
            #step_size = batch_step_size * gamma**(1. / n_batches_per_epoch)
            #step_size = batch_step_size * (1.14*(10**14))**(1. / n_batches_per_epoch) 
            #step_size = batch_step_size+1e-3
            #print("step_size", step_size)
            # only do the check if the gradient norm is big enough
            with torch.no_grad():
                if grad_norm >= 1e-8:
                    # check if condition is satisfied
                    found = 0
                    step_size_old = step_size
                    #print("start of search")
                    #for e in range(100):
                    for e in range(10):
                        # try a prospective step
                        ut.try_sgd_update(params, step_size, params_current, grad_current)

                        # compute the loss at the next step; no need to compute gradients.
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1
                        number_of_iterations+=1
                        # =================================================
                        # Line search
                        if group['line_search_fn'] == "armijo":
                            armijo_results = ut.check_armijo_conditions_acc(step_size=step_size,
                                                        step_size_old=step_size_old,
                                                        c=group['c'],
                                                        beta_b=group['beta_b'],
                                                        images=images,
                                                        labels=labels,
                                                        model=model)
                            found, step_size, step_size_old = armijo_results
                            #print(step_size, found)
                            if found == 1:
                                break
                        
                        '''elif group['line_search_fn'] == "goldstein":
                            goldstein_results = ut.check_goldstein_conditions(step_size=step_size,
                                                                    loss=loss,
                                                                    grad_norm=grad_norm,
                                                                    loss_next=loss_next,
                                                                    c=group['c'],
                                                                    beta_b=group['beta_b'],
                                                                    beta_f=group['beta_f'],
                                                                    bound_step_size=group['bound_step_size'],
                                                                    eta_max=group['eta_max'])

                            found = goldstein_results["found"]
                            step_size = goldstein_results["step_size"]

                            if found == 3:
                                break'''
                
                    # if line search exceeds max_epochs
                    if found == 0:
                        ut.try_sgd_update(params, 1e-6, params_current, grad_current)

            # save the new step-size
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return loss, number_of_iterations, step_size, step_size_old

