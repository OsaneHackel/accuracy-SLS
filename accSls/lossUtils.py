#TODO: 
#implementieren von accSls aud dem Loss
#Vergleichen der Steigungsberechnung mit accSls (Slope) und mit der torch.autograd.grad Funktion (auf dem Loss)
#vergleichen der performance von accSls auf dem Loss und deren sls

import copy
import torch


def computeSlope_vector_loss(model, images, labels, eps=1e-4):
        tetha=torch.nn.utils.parameters_to_vector(model.parameters())
        #v is gradient of the loss
        v=[p.grad for p in model.parameters()]
        v=torch.nn.utils.parameters_to_vector(v)
        v= v/torch.linalg.vector_norm(v)

        torch.nn.utils.vector_to_parameters(tetha+eps*v, model.parameters())
        loss1 =l(model, images, labels)
        
        #print("loss1_vector: ", loss1)
        torch.nn.utils.vector_to_parameters(tetha-eps*v, model.parameters())
        loss2 = l(model, images, labels)
        
        #print("loss2_vector: ", loss2)
        #print("difference_vector: ", loss1-loss2)
        #bring model back to original state
        torch.nn.utils.vector_to_parameters(tetha, model.parameters())
        slope = (loss1-loss2)/(2*eps)
        return slope


#softmax_loss they use
def l(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def check_armijo_conditions_loss(step_size, step_size_old,
                       c, beta_b, images, labels, model): 
    found = 0

    #create copy of the model with changed parameters(depending on step_size)
    ami_model = copy.deepcopy(model)
    ami_model.train()
    old_parameters = copy.deepcopy(list(model.parameters()))
    for param_ami, param_old, param_new in zip(ami_model.parameters(), old_parameters, model.parameters()):
                            param_ami.data = param_old - step_size * param_new.grad #TODO: which stepsize should I use?
    #compute accuracy of the new model
    loss_act=l(ami_model, images, labels)
    loss_old = l(model, images, labels)
    #slope = computeSlope_vector_loss(model, images, labels)
    #####################original version of break condition#####################
    #break_condition =  loss_act-(loss_old + (step_size) * c * slope**2) 
    #if this is smaller than 0, the condition is satisfied!!!

    #####################test version of break condition##########################################
    grad_norm=compute_grad_norm([p.grad for p in model.parameters()]) #TODO:rigth input??
    break_condition = loss_act- \
          (loss_old + (step_size) * c * grad_norm**2) 
    
    #####################test version of break condition other grad norm computation##########################################
    '''for group in model.parameters():
        params=group["params"]
        grad_current=ut.get_grad_list(params)
        grad_norm=compute_grad_norm(grad_current)''' #sehr strange????

    break_condition = loss_act- \
          (loss_old + (step_size) * c * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old

#computes the gradient norm of the loss
def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm