import numpy as np
from time import time
import torch
import os

def train_model(n_epochs, model, train_loader, criterion, optimizer, device = 'cuda', scheduler=None):
    # define training loop
    train_loss_min = np.Inf 
    train_loss_list = list()
    
    # model = model.to(device)

    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        total_train = 0
        # train model
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() 
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            total_train += target.shape[0]

        if scheduler != None:
            # update scheduler
            scheduler.step()

        # compute average loss
        train_loss /= total_train
        train_loss_list.append(train_loss)
        
        # save best model
        if train_loss <= train_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(train_loss_min, train_loss))

            if not os.path.isdir('model_trained'):
                os.mkdir('model_trained')

            torch.save(model.state_dict(), f'./model_trained/m{int(time())}.pth') 
            train_loss_min = train_loss

        print('Model Trained: Epoch: {}/{} \tTrain Loss: {:.6f}' .format(epoch, n_epochs, train_loss))
    

    #print('Time elapsed: {} hours, model training please drive well'.format((end - start) / 3600.0))
    return True 