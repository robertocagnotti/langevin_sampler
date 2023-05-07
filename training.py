#from tqdm import tqdm
import torch
from torch import nn

def train(model:nn.Module,
          z:torch.Tensor,
          img:torch.Tensor,
          num_epochs:int=10,
          criterion=None,
          regularization='l2',
          w_decay=0.01,
          optimizer=None,
          device:str="cpu",
          save_path:str="outputs/model.pt",
          verbose=True):
    
    '''
    inputs:
        model: the deepdecoder network
        z: input tensor of the network
        img: the noisy image
        num_epochs: number of training iterations
        criterion: loss function
        regularization: must be 'l2' or 'l1'
        w_decay: weight decay used for regularization
                 must be given here only if regularization = 'l1'
                 in the 'l2' case it must be given as input in the optimizer
        optimizer: optimiser
        save_path: path where to save the trained model
        verbose: if True, prints informations on the training
        
    output is a dictionary containing:
        train_loss: list of the losses at each iteration    
    '''
    
    # Logs for the training loss
    log_train_loss = list()

    # Set the device for the model
    model.to(device)
    if verbose:
        print(f"Training on device: {device}")

    for epoch in range(num_epochs):

        train_loss = 0.0

        # Set the model to training mode
        model.train()
        
        # Move the images and labels to the device
        image = img.to(device)
        input_tensor = z.to(device)

        # Forward pass
        output = model(input_tensor)

        # Compute the loss
        loss = criterion(output, image)
        if regularization == 'l1':
            l1_norm = sum(torch.sum(torch.abs(p)) for p in model.parameters())
            loss = loss + w_decay/2 * l1_norm

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update loss
        train_loss += loss.item()

        # Update the weights
        optimizer.step()

        # Log the training loss
        log_train_loss.append(train_loss)
        
        if verbose:
            print('Iteration:', epoch, end='\r')
        
    training_log = {
        "train_loss": log_train_loss,
    }
    
    # Save model
    torch.save(model.state_dict(), save_path)
    
    if verbose:
        print('\nTraining finished')
    return training_log