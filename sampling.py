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
                p.data = p.data - 0.5*group['lr']**2*p.grad.data + group['lr']*eps

def get_gradient_vector(model, device):
    grad = torch.empty(1).to(device)
    for param in model.parameters():
        grad_new = torch.flatten(param.grad)
        grad = torch.cat((grad,grad_new))
    grad = grad[1:]
    return grad

def loss_function(y,f_theta,theta,prior,sigma,nu):
    mse = nn.MSELoss(reduction='sum')
    mse1 = mse(y,f_theta)
    if prior == 'l2':
        regul = torch.linalg.vector_norm(theta)**2
    elif prior == 'l1':
        regul = torch.sum(torch.abs(theta))
    loss = 0.5/sigma**2*mse1 + 0.5/nu**2*regul
    loss_for_plot = mse1.item()
    return loss, loss_for_plot

def acceptance_ratio(img,output1,output2,theta1,theta2,mu_theta1,mu_theta2,
                     prior,sigma,nu,lr):
    mse = nn.MSELoss(reduction='sum')
    if prior == 'l2':
        r = torch.exp(0.5/sigma**2*(mse(img,output1)-mse(img,output2)) + \
                      0.5/nu**2*(torch.linalg.vector_norm(theta1)**2 - \
                                 torch.linalg.vector_norm(theta2)**2) + \
                      0.5/lr**2*(mse(theta2,mu_theta1)-mse(theta1,mu_theta2)))
    elif prior == 'l1':
        a = torch.sum(torch.abs(theta1))
        b = torch.sum(torch.abs(theta2))
        r = torch.exp(0.5/sigma**2*(mse(img,output1)-mse(img,output2)) + \
                      0.5/nu**2*(a-b) + \
                      0.5/lr**2*(mse(theta2,mu_theta1)-mse(theta1,mu_theta2)))
    r = r.item()
    if r >= 1:
        r = 1
    return r

def langevin_sample(model:nn.Module,
                    z:torch.Tensor,
                    img:torch.Tensor,
                    num_samples_wanted:int=10,
                    prior='l2',
                    lr=0.01,
                    sigma=0.707106781,
                    nu=7.071067812,
                    device:str="cpu",
                    debug:bool=False,
                    verbose=True):
    '''
    inputs:
        model: the deepdecoder network
        z: input tensor of the network
        img: the noisy image
        num_samples_wanted: number of samples to successfully ectract
        prior: must be 'l2' for Gaussian prior or 'l1' for Laplace prior
        lr: learning rate of MALA
        sigma: standard deviation of likelihood, default 1/sqrt(2)
        nu: standard deviation of prior, default 10/sqrt(2)
        device: device to perform to iterations on, e.g. 'cuda', 'cpu', 'mps'
        debug: if True, prints the acceptance probabilities at each step
        verbose: if True, prints informations on number of samples extracted
    
    output is a dictionary containing:
        thetas: a list containing the last 20% the samples extracted
                (each sample is a vector of parameters)
        log_loss: a list containing the loss of the samples extracted
        log_loss_total: a list containing the loss of at each iteration
                        (including of the samples rejected)
        ratios: a list containing the acceptance probabilities of all the steps
    '''
    
    # Logs for the samples
    log_loss = list()
    log_loss_total = list()
    thetas = list()
    ratios = list()

    # Set the device for the model
    model.to(device)
    if verbose:
        print(f"Sampling on device: {device}")
    optimizer = MyOptimizer(model.parameters(),lr=lr)
    
    # Get the samples
    max_iter = 1000000
    iteration = 0
    sample_count = 0
    t = 0
    while sample_count < num_samples_wanted:
        # Stop algorithm if max_iter is reached
        if iteration > max_iter:
            break

        # Set the model to training mode
        model.train()
        optimizer.zero_grad()
        
        # Move the images and labels to the device
        img = img.to(device)
        z = z.to(device)
        
        # Save current parameters
        theta = nn.utils.parameters_to_vector(model.parameters())
        theta_copy = theta.detach()

        # Compute loss
        output1 = model(z)
        loss, _ = loss_function(img,output1,theta,prior,sigma,nu)

        # Backward pass
        loss.backward()
        
        # Compute mu_theta
        grad = get_gradient_vector(model, device)
        mu_theta = theta_copy - 0.5*lr**2*grad
        
        # Update temporarily weights
        optimizer.step()
        optimizer.zero_grad()
        theta_star = nn.utils.parameters_to_vector(model.parameters())
        theta_star_copy = theta_star.detach()
        
        # Compute mu_theta_star
        output2 = model(z)
        loss, loss_for_plot = loss_function(img,output2,theta_star,prior,sigma,nu)
        loss.backward()
        grad = get_gradient_vector(model, device)
        mu_theta_star = theta_star_copy - 0.5*lr**2*grad
        
        # Compute acceptance ratio
        r = acceptance_ratio(img,output1,output2,theta_copy,theta_star_copy,mu_theta,mu_theta_star,
                         prior,sigma,nu,lr)
        ratios.append(r)
        if debug:
            print('r:',r)
        
        # Update the model
        u = torch.rand(1).to(device)
        if u <= r:
            nn.utils.vector_to_parameters(theta_star_copy,model.parameters())
            # Save last 20% of the samples
            if t>int(0.8*num_samples_wanted):
                theta_log = theta_star_copy
                thetas.append(theta_log)
            sample_count+=1
            log_loss.append(loss_for_plot)
            log_loss_total.append(loss_for_plot)
            t+=1
        else:
            nn.utils.vector_to_parameters(theta_copy,model.parameters())
            log_loss_total.append(loss_for_plot)
                
        iteration+=1
        
        if not debug:
            if verbose:
                print('Samples extacted:', sample_count,
                      '    Total iterations:',iteration, end='\r')    
    log_samples = {
        "thetas": thetas,
        "losses": log_loss,
        "total_losses": log_loss_total,
        "ratios": ratios
    }
    if verbose:
        print('\nSampling finished')
        print('Acceptance rate:', sample_count/iteration)
    return log_samples