import torch
from torch import nn
import numpy as np

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)
  
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                device = p.get_device()
                if device == -1:
                    device = 'cpu'
                if p.grad is None:
                    continue
                size = p.data.shape
                eps = torch.randn(size).to(device)
                p.data = p.data + 0.5*group['lr']**2*p.grad.data + group['lr']*eps
    
def langevin_sample(theta_zero, num_samples_wanted:int=10, lr=0.01, device:str="cpu",
                   dim:int=2):
    
    # Logs for the samples
    thetas = list()
    
    theta = theta_zero.clone().detach().requires_grad_(True)
    
    # Set the device for the model
    optimizer = MyOptimizer([theta],lr=lr)
    
    # Get the samples
    sample_count = 0
    iteration=0
    while sample_count < num_samples_wanted:

        # Set the model to training mode
        optimizer.zero_grad()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                theta = p
                
        # Compute the loss
        mu = torch.zeros(dim, device = device)
        sigma = torch.eye(dim, device = device)
        loss = - 0.5 * (theta - mu) @ sigma @ (theta - mu)

        # Backward pass
        loss.backward()
        
        mu_theta = theta + 0.5*lr**2*theta.grad
        
        # Update weights
        optimizer.step()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                theta_star = p
        #print('theta star',theta_star)
        
        loss = - 0.5 * (theta-mu) @ sigma @ (theta - mu)
        loss.backward()
        
        mu_theta_star = theta_star + 0.5*lr**2*theta_star.grad
        #print('mu theta star',mu_theta_star)

        r = (-0.5 * (theta_star-mu)@sigma@(theta_star-mu) \
             -0.5/lr**2 * (theta-mu_theta_star)@sigma@(theta-mu_theta_star) \
             +0.5 * (theta-mu)@sigma@(theta-mu) \
             +0.5/lr**2 * (theta_star-mu_theta)@sigma@(theta_star-mu_theta)
            )
        r = r.item()
        if r>=0:
            r=0
        r = np.exp(r)
        u = torch.rand(1).to(device)
        if u>r:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p = theta.detach()
                    p.requires_grad = True
        else:
            sample_count+=1
            thetas.append(theta_star.detach().cpu().numpy())
            
        iteration += 1

        print('Samples extacted:', sample_count, '    Total iterations:',iteration, end='\r')
      
    log_samples = {
        "thetas": thetas
    }
    print('\nSampling finished')
    return log_samples