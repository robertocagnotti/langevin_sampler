import torch
from torch import nn, optim
import numpy as np
from PIL import Image

def get_loss_function():
    return nn.MSELoss(reduction='sum')

def get_optimizer(network, lr=0.001, weight_decay=0):  
    return optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

def reconstruct(z,model,device='cpu'):
    '''
    Takes the model and the input tensor and return the reconstructed image.
    '''
    z = z.to(device)
    model = model.to(device)
    recon_img = model(z).cpu()
    recon_img = recon_img.data.numpy()[0]
    out_img = np.clip(recon_img.transpose(1, 2, 0),0,1)*255
    out_img = out_img.astype('uint8')
    if out_img.shape[2]==1:
        out_img = np.squeeze(out_img)
        im = Image.fromarray(out_img,mode='L')
    else:
        im = Image.fromarray(out_img)
    return im

def load_img(img_path):
    '''
    Takes the input image and returns it as a tensor of shape (1,k_out,xdim,ydim),
    where k_out = 1 or 3 (gray-scale, RGB).
    '''
    img_pil = Image.open(img_path)
    ar = np.array(img_pil)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    img_np = ar.astype(np.float32) / 255.
    img_tens = torch.from_numpy(img_np)
    img_tens = torch.unsqueeze(img_tens,0)
    return img_tens

def load_img_np(img_path):
    '''
    Takes the input images and returns it as a numpy array.
    '''
    img_pil = Image.open(img_path)
    ar = np.array(img_pil)
    if len(ar.shape) == 1:
        ar = ar[None, ...]
    img_np = ar.astype(np.float32) / 255.
    return img_np

def psnr(x_hat,x_true,maxv=1.):
    '''
    Computes PSNR between x_hat and x_true.
    '''
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

def get_noisy_img(img,sig=30,noise_same=False):
    '''
    Takes the input image and returns a noisy version of it as numpy array.
    The noise level is decided by sig.
    '''
    sigma = sig/255.
    if noise_same: # add the same noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape[1:])
        noise = np.array([noise]*img_np.shape[0] )
    else: # add independent noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape)

    img_noisy_np = np.clip(img_np + noise , 0, 1).astype(np.float32)
    return img_noisy_np