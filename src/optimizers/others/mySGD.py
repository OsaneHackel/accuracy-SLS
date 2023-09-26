import torch
import copy
from torch.optim import Optimizer

class MySGD(Optimizer):
    def __init__(self, params):
        print("hier")
        self.lr=0.1
        defaults = dict(lr=0.1)
        super().__init__(params, defaults) #super(CocobBackprop, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        #deriv = loss.backward()
        #self.zero_grad()
        loss.backward()
        #pr = True
        #need to put this somehow into the code
        #optimizer.zero_grad()
        for group in self.param_groups:
            lr = group['lr']
            #print(group["params"])
            for p in group['params']:
                #if p.grad is None:    
                #    continue
                #print(p.grad)
                
                grad = p.grad
                #if grad is not None:
                    #print("hi")
                    #print(lr*grad)
                p.data -= lr * grad

            #params = group["params"]
            #params_current = copy.deepcopy(params)
            #group["params"]=params_current - 0.1* deriv

        return loss
        #params_current = copy.deepcopy(self.params)

    """   if pr:
                    print(p)
                    print(p.grad)

                if p.grad is None:
                    continue
                #d_p = p.grad
                #p.add_(d_p, -0.1)
                if pr: 
                    print(p)
                    pr = False
                #new_val = p - 0.1 * p.grad
                new_val = 
                p.copy_(new_val)"""
        
""" def step(self, closure): 
        #old_params=params not necessary
        #loss = closure()
        #loss = None
        #if closure is not None:
        #loss = closure()
        #loss.backward()
        #output = model(input)
        #loss = loss_fn(output, target)
        #model.zero_grad()
        #grad = loss.backward()
        print('lr')
        for i, param in enumerate(params):
            d_p = closure()
            param.add_(d_p, -0.1)
                    
        return d_p"""
"""
    class mySGD(torch.optim.Optimizer):
        def _init_(self, params, n_batches_per_epoch = 100, step_size= 0.01):
        defaults = dict (
            n_batches_per_epoch=n_batches_per_epoch,
            step_size=step_size
        )
        super()._init_(params, defaults)
        self.state['step_size']=0.01


    def step(self, model, input, target, loss_fn, params):
        old_params=params
        output = model(input)
        loss = loss_fn(output, target)
        model.zero_grad()
        grad = loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                new_val = p - step_size * p.grad
                p.copy_(new_val)
"""