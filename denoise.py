import numpy as np
import torch
from torch import nn
from PIL import Image
from time import time
import os

from deepdecoder import DeepDecoder
from training import train
from sampling import langevin_sample
from utils import *


def DENOISE(noisy_img,epochs_pretrain,num_samples,epochs_retrain,name,device='cpu'):
    
    '''
    This functions does the same as illustraded in the notebook 'results.ipynb'
    
    inputs:
        noisy_img: input noisy image
        epochs_pretrain: number of iterations in the initial training
        num_samples: number of samples to extract with MALA
        epoch_retrain: number of iterations for each training starting from the samples extracted
        name: name representing the image (could be simply an ID)
        device: device available: 'cuda', 'cpu', 'mps',...
    
    outputs:
        The function doesn't return outputs, but several elements will be saved in the folder
        outputs/'name':
            initial_train_reconstruction.png: reconstruction of the image after initial training
            initial_train_model.pt: state_dict of the initially trained model
            retrained_models: folder containing all the state_dict of the retrained models
            AVG_reconstruction: average of the reconstructrions after retraining
    
    Note all the hyperparameters are set to our default. One might need to tweaks some of them depending on the input image.
    Especially important is the learning rate of the langevin_sample function:
        - if lr is too high, samples are never accepted
        - if lr is too low, the algorithm learns too slowly
    '''
    
    ### DATA LOADING AND HYPERPARAMETERS ###
    
    k = 64 # number of channels of the DeepDecoder
    y = noisy_img # Tensor of shape (1,k_out,512,512)
    k_out = y.shape[1]
    dim = y.shape[2]
    model = DeepDecoder(k=k,k_out=3)
    z = torch.load(f'data/z{k}.pt')

    ### INITIAL TRAINING ###

    path = f'outputs/{name}'
    os.makedirs(f'{path}', exist_ok=True)
    save_path = f'{path}/initial_train_model.pt'

    loss_fct = get_loss_function()
    optimizer = get_optimizer(model,lr=0.01, weight_decay=0.02)

    log_loss = train(model,
                     z,
                     img=y,
                     num_epochs=epochs_pretrain,
                     criterion=loss_fct,
                     optimizer=optimizer,
                     device=device,
                     save_path=save_path,
                     verbose=False)
    
    recon_init_train = reconstruct(z,model,device=device)
    recon_init_train.save(f'{path}/initial_train_reconstruction.png')
    
    ### SAMPLING ###

    log_samples = langevin_sample(model=model,
                          z=z,
                          img=y,
                          num_samples_wanted=num_samples,
                          prior='l2',
                          lr=0.0015,
                          sigma=0.707106781,
                          nu=7.071067812,
                          device=device,
                          debug=False,
                          verbose=False)

    thetas = log_samples['thetas']

    #Select samples
    thetas_cut = thetas[:]
    new_thetas = list()
    for i, theta in enumerate(thetas_cut):
        if i%4==0:
            new_thetas.append(theta)

    ### SECOND TRAINING ###

    os.makedirs(f'{path}/retrained_models', exist_ok=True)   
    for i, theta in enumerate(new_thetas):
        nn.utils.vector_to_parameters(theta,model.parameters())
        optimizer = get_optimizer(model,lr=0.01, weight_decay=0.02)
        save_path = f'{path}/retrained_models/model_{i:04d}.pt'
        log_loss = train(model,
                         z,
                         img=y,
                         num_epochs=epochs_retrain,
                         criterion=loss_fct,
                         optimizer=optimizer,
                         device=device,
                         save_path=save_path,
                         verbose=False)

    ### COMPUTE OUTPUT ###

    if k_out == 1:
        img_sum = np.zeros((dim,dim))
    elif k_out == 3:
        img_sum = np.zeros((dim,dim,3))

    for i in range(len(new_thetas)):
        model.load_state_dict(torch.load(f'{path}/retrained_models/model_{i:04d}.pt',
                                         map_location=torch.device(device)))
        img = reconstruct(z,model,device=device)
        img = np.array(img)
        img_sum += img

    img_avg = img_sum/len(new_thetas)
    img_avg = img_avg.astype('uint8')
    img_avg = Image.fromarray(img_avg)
    if k_out == 1:
        img_avg = img_avg.convert('L')
    elif k_out == 3:
        img_avg = img_avg.convert('RGB')
    img_avg.save(f'{path}/AVG_reconstruction.png')
