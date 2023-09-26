import torch
import copy
import numpy as np
import contextlib
import time
from src import metrics


eps = 1e-6

#definition of different target functions
def accuracy_f(model, images, labels, slope):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()
    return acc

def weak_accuracy_f(model, images, labels, slope):
    logits =model(images)
    p=torch.softmax(logits,dim=1)
    p2 = p.clone()
    p2.scatter_(1, labels[:, None], 0.0)
    dp=p.gather(dim=1, index=labels[:, None])-torch.max(p2, dim=1,keepdim=True)[0]
    acc = torch.sigmoid(slope*dp).mean()
    return acc


def loss2_f(model, images, labels, backwards=False, slope=3):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def loss_f(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss



###################################### TODO#############################################
def check_armijo_conditions_basic(step_size, c, beta_b, images, labels, model, loss_grad, grad_norm, eps=1e-4, target_f = None, minimize=True, slope=3):
    if target_f is None:
        target_f = loss_f

    #print('check_armijo_conditions_basic_acc')
    found = 0
    forw=0
    f_null = target_f(model, images, labels, slope=slope)  
    forw+=1
    #normed_grad = loss_grad / grad_norm
    w = torch.nn.utils.parameters_to_vector(model.parameters())
    
    ###old
    #torch.nn.utils.vector_to_parameters(w-eps*normed_grad, model.parameters())
    #f_eps = target_f(model, images, labels, slope=slope)
    #forw+=1

    #######TEST
    torch.nn.utils.vector_to_parameters(w-eps*loss_grad, model.parameters())
    forw+=1
    f_eps_unnormed=target_f(model, images, labels, slope=slope)
    #slope_unnormed=(f_null-f_eps_unnormed)/eps
    ##################
    torch.nn.utils.vector_to_parameters(
        w-step_size*loss_grad, model.parameters())
    f_lr_star = target_f(model, images, labels, slope=slope)
    forw+=1
    torch.nn.utils.vector_to_parameters(w, model.parameters())
    
    if minimize:
        #delta_norm= f_null -f_eps  
        delta=f_null-f_eps_unnormed      
    else:
        #delta_norm =  f_eps - f_null  
        delta= f_eps_unnormed-f_null

    #if delta_acc < 0:
    #   raise ValueError("loss_eps > f_null, eps might be too large")
    if delta < 1e-3 * eps:
        #print(f'very small delta acc: {delta:.3e}  eps:{eps:.2e} f_eps:{f_eps_unnormed:.3f}  f_null:{f_null:.3f}')
        #slope = 0
        #slope_normed=0
        slope_unnormed=0
        #return 1,step_size, forw  #Test Nr.1 try, what happens if I do nothing if slope is 0
        #return 1,step_size*beta_b,forw #Test Nr.2, try to avoid the expoential growth of Test Nr.1
    else:
        #slope_normed = delta_norm/eps
        slope_unnormed=delta/eps
        #calc_diff = (slope-grad_norm).abs()
        #if calc_diff > 1e-2:
            #print(f"diff: {calc_diff:.3e} - grad_norm: {grad_norm:.3e}  slope: {slope:.3e}  delta_loss: {delta:.3e}")
    #slope_normed=(f_null-f_eps)/delta
    
    #change sign depending on: loss or accuracy
    if minimize:
        #break_condition = f_lr_star <= (f_null - (step_size) * c * slope**2)
        break_condition = f_lr_star <= (f_null - (step_size) * c * slope_unnormed)
    else:
        #break_condition =acc_lr_star >=  (acc_null + (step_size) * c * slope**2) 
        #break_condition = f_lr_star >= (f_null + step_size*c*slope_normed*grad_norm) #new rule for accuracy based line serach 
        break_condition = f_lr_star >= (f_null + step_size*c*slope_unnormed)

    if (break_condition): #eps area 
        found = 1
    else:
        step_size = step_size * beta_b
    #TEST
    #print(f"normed slope*grad_norm: {(slope_normed*grad_norm):.5f}------|------slope unnorm: {slope_unnormed:.5f}")#-----|-----normed slope: {slope_normed:.3f}-----|-----unnormed slope: {slope_unnormed:.3f}")

    return found, step_size, forw


def lr_scheduler(option, batch_step_size, gamma, n_batches_per_epoch):
    if option == 'constant':
        step_size = batch_step_size + gamma

    # multiplies the stepsize with a constant and weights it with the number of batches per epoch
    elif option == 'multiplicative':
        # TODO change 467 to n_batches_per_epoch back
        step_size = batch_step_size * gamma**(1. / n_batches_per_epoch)

    # multiplies the stepsize with a constant (to increse it by x percent)
    elif option == 'scaling':
        step_size = batch_step_size * gamma

    # print(f"new: {step_size:.3e}, old: {batch_step_size:.3e}")
    return step_size


def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)
        # print(f"gamma reset to {step_size}")

    elif reset_option == 2:
        # print("init_step_size", init_step_size)
        step_size = init_step_size

    return step_size


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current


def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]


def sa(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()


@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)
