#TODO: 
#implementieren von accSls aud dem Loss
#Vergleichen der Steigungsberechnung mit accSls (Slope) und mit der torch.autograd.grad Funktion (auf dem Loss)
#vergleichen der performance von accSls auf dem Loss und deren sls



#importieren der verschiedenen Module
#import accSls.utils as ut
import copy
import torch
from decimal import Decimal, getcontext
import time
#from bigfloat import *

getcontext().prec = 100
#checks if the different functions compute the same loss slope for a certain state of the model
def test_if_all_equal(model, images, labels, eps):
        time_basic=time.time()
        a=computeSlope_basic_loss(model, images, labels, eps)
        time_basic=time.time()-time_basic
        time_2_points=time.time()
        b=computeSlope_2_points_loss(model, images, labels, eps)
        time_2_points=time.time()-time_2_points
        time_vector=time.time()
        c=computeSlope_vector_loss(model, images, labels, eps)
        time_vector=time.time()-time_vector
        #computation of the loss slope with torch.autograd.grad
        v=[p.grad for p in model.parameters()]
        v=torch.nn.utils.parameters_to_vector(v)
        d=torch.linalg.vector_norm(v)
        e=torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)
        #loss = ut.metrics.softmax_loss(model, images, labels, backwards=False)
        #loss.backward()
        #gradient=ut.torch.autograd.grad(loss, model.parameters(), create_graph=True)
        #print(gradient)      
        #compute norm with torch
        
        #d=gradient/norm
                                     
        
        print('--------------------------------------------------------------------------------------------------------------------')
        print("Test if all functions compute the same loss slope")
        print("basic: ", a)
        #print(f"computational time basic: {time_basic}")
        print("2 points: ", b)
        #print(f"computational time 2 points: {time_2_points}")
        
        #print("basic: ", a)
        print("vector: ", c)
        #print(f"computational time vector: {time_vector}")
        print("torch_vector: ", Decimal(d.item()))
        print("torch_clip: ", Decimal(e.item()))
        return a==b==c==d==e


        
        



#compute slope on the loss############################################################################################################################
def computeSlope_basic_loss(model, images, labels, eps=1e-4):

    #direction of change is the gradient of the loss
    #loss = metrics.softmax_loss(model, images, labels, backwards=False)
   
    #creatae copy of model with changed parameters (for p1/p2)
    p1_model = copy.deepcopy(model)
    p1_model.eval() 
    p2_model = copy.deepcopy(model)
    p2_model.eval()
    for param_p1, param_p2, param_curr in zip(p1_model.parameters(),p2_model.parameters(), model.parameters()):
                        param_p1.data = param_curr + eps * param_curr.grad
                        param_p2.data = param_curr - eps * param_curr.grad 

                        #TODO log norm
    norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)    
    norm = Decimal.from_float(norm.item())              

    #compute loss for p1 and p2
    loss1 = l(p1_model, images, labels)
    loss1=Decimal.from_float(loss1.item())
    print("loss1_basic: ", loss1)
    loss2 = l(p2_model, images, labels)
    loss2=Decimal.from_float(loss2.item())
    print("loss2_basic: ", loss2)
    print("norm_basic: ", norm)
    print("difference_basic: ", loss1-loss2)
    
    #slope = Decimal((loss1-loss2)/(Decimal(2)*Decimal(eps)*norm))
    slope = Decimal((loss1-loss2)/(Decimal(2)*Decimal(eps)))

    return slope #TODO: needs to have positive value -> if slope is negative, change direction of change or get new batch??

def computeSlope_2_points_loss(model, images, labels, eps=1e-4):
    #with precision(64):
        p1_model = copy.deepcopy(model)
        p1_model.eval()
        for param_p1, param_curr in zip(p1_model.parameters(), model.parameters()):
                        param_p1.data = param_curr + eps * param_curr.grad

        norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)
        norm=Decimal(norm.item())
        loss1 =l(p1_model, images, labels)
        loss1=Decimal(loss1.item())
        print("loss1_2_points: ", loss1)
        loss2 =l(model, images, labels)
        loss2=Decimal(loss2.item())
        print("loss2_2_points: ", loss2)
        print("norm_2_points: ", norm)
        

        print("difference_2_points: ", loss1-loss2)
        #slope = Decimal((loss1-loss2)/(Decimal(eps)*norm))
        slope = Decimal((loss1-loss2)/Decimal(eps))

        return slope


def computeSlope_vector_loss(model, images, labels, eps=1e-4):
        tetha=torch.nn.utils.parameters_to_vector(model.parameters())
        #v is gradient of the loss
        v=[p.grad for p in model.parameters()]
        v=torch.nn.utils.parameters_to_vector(v)
        #v= v/torch.linalg.vector_norm(v)

        torch.nn.utils.vector_to_parameters(tetha+eps*v, model.parameters())
        loss1 =l(model, images, labels)
        loss1=Decimal(loss1.item())
        print("loss1_vector: ", loss1)
        torch.nn.utils.vector_to_parameters(tetha-eps*v, model.parameters())
        loss2 = l(model, images, labels)
        loss2=Decimal(loss2.item())
        print("loss2_vector: ", loss2)
        print("difference_vector: ", loss1-loss2)
        #bring model back to original state
        torch.nn.utils.vector_to_parameters(tetha, model.parameters())
        slope = Decimal((loss1-loss2)/(2*Decimal(eps)))
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
    slope = computeSlope_vector_loss(model, images, labels)
    break_condition =  (loss_old + (step_size) * c * slope**2) - loss_act
    #if this is smaller than 0, the condition is satisfied!!!
    #TODO

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old