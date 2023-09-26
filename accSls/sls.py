import torch
import copy
import time
import accSls.utils as ut


class Sls(torch.optim.Optimizer):
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
                    what works: mnist: constant with gamma 0.001
            
        searchBatch: "same","small", "big" speciefies the batch on which accSls makes line search. if big is choosen, lr is only updated eberz 1000th batch
        version: "loss" and "accuracy": determines on which metric linesearch is conducted
    """


    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1.0,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2,
                 eta_max=10,
                 lr_step_schedule='multiplicative'):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        c=c,
                        beta_b=beta_b,
                        gamma=gamma,
                        eta_max=eta_max,
                        lr_step_schedule=lr_step_schedule)
        super().__init__(params, defaults)       

        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

    def step(self, closure, images, labels, model, searchBatch, train_set, number_of_steps, normalization, version, slope):
        maximus = 100 #defines how often the model is updated without line search
        # deterministic closure
        step_size = self.state['step_size']
        #print("step size: ", step_size) #diag
        number_of_iterations = 0
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()
        # get loss and compute gradients
        #crit=torch.nn.CrossEntropyLoss(reduction="none")
        #loss2=crit(model(images), labels.view(-1))
        #print("loss2: ", torch.topk(loss2, 3))
        loss = closure_deterministic()
        loss.backward()
        self.state['n_backwards'] += 1
        self.state['n_forwards'] += 1

        #increase the learning rate exept we do maximuis approach
        if not(searchBatch == "big"):
            step_size = ut.lr_scheduler(option=self.defaults['lr_step_schedule'],batch_step_size=step_size,
                    n_batches_per_epoch = self.defaults['n_batches_per_epoch'],
                    gamma = self.defaults['gamma'])
        
        #draw a batch on which the line search is performed
        if searchBatch == "same":
            images2, labels2 = images, labels
        elif searchBatch == "small":
            images2, labels2 = giveBatch(train_set, 128)
            images2, labels2=images2.cuda(), labels2.cuda()
            #print('draw other batch')
        elif searchBatch == "big":
            if number_of_steps % maximus == 0:
                images2, labels2 = giveBatch(train_set, 1024)
                images2, labels2=images2.cuda(), labels2.cuda()
            else:
                pass #TODO should update the model here (but without line search for lr)
        # only do the check if the gradient norm is big enough
        
        with torch.no_grad():
            # compute the gradient norm
            loss_grad=[p.grad for p in model.parameters()]
            #print(f"gradients {loss_grad}")
            loss_grad=torch.nn.utils.parameters_to_vector(loss_grad)
            grad_norm = torch.norm(loss_grad)    #Attention! when to use normalized and when not?
            #print("gradient norm: ", grad_norm) #diag
            ##
            if normalization:
                loss_grad = loss_grad / grad_norm
            #print("grad norm: ", grad_norm)
            step_size_old = step_size
            if not(searchBatch == "big" and number_of_steps % maximus != 0):
                if searchBatch == "big":
                    print("performing line search:")
                if grad_norm >= 1e-8:
                    # check if condition is satisfied
                    found = 0
                    #print("start of search")
                    #for e in range(100):
                    for e in range(100):
                        number_of_iterations+=1
                        
                        #==================================================
                        # they update parameters early
                        #w = torch.nn.utils.parameters_to_vector(model.parameters())
                        #torch.nn.utils.vector_to_parameters(w - step_size * loss_grad, model.parameters())

                        # =================================================
                        # Line search
                        
                        minimize = True
                        if version == "accuracy":
                            target_f = ut.accuracy_f
                            minimize = False
                        elif version == "weak_accuracy":
                            target_f = ut.weak_accuracy_f
                            minimize = False
                        elif version == 'loss':
                            target_f = ut.loss2_f
                            minimize = True
                        else:
                            raise ValueError(f'Unknown linesearch version: {version}') 
                        
                        armijo_results = ut.check_armijo_conditions_basic(step_size=step_size,
                                                    c=self.defaults['c'],
                                                    beta_b=self.defaults['beta_b'],
                                                    images=images2,
                                                    labels=labels2,
                                                    model=model,
                                                    loss_grad=loss_grad,
                                                    grad_norm=grad_norm,
                                                    target_f=target_f,
                                                    minimize=minimize,
                                                    slope=slope)
                                              
                        found, step_size, forw= armijo_results
                        #print(f"armijo results: found {found}, step size {step_size}") #diag
                            #print(step_size, found)
                        self.state['n_forwards'] += forw
                        if found == 1:
                            #print("found")
                            break
                
                    # if line search exceeds max_epochs -> no learning happens
                    if found == 0:
                        print("line search failed")
            else:
                found = 1
            # =================================================
            # Update parameters
            w = torch.nn.utils.parameters_to_vector(model.parameters())
            torch.nn.utils.vector_to_parameters(w - step_size * loss_grad, model.parameters())
            
            # save the new step-size
            #print(step_size)
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return loss, number_of_iterations, step_size, step_size_old, grad_norm, loss
    

def giveBatch(train_set, n):
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=n, shuffle=True,
        pin_memory=False)
    images, labels = next(iter(train_loader))
    return images, labels


    