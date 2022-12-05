import numpy as np
from time import time
import torch
import os

def train_model(n_epochs, model, train_loader, test_loader, criterion, optimizer, device = 'cpu', scheduler=None):
    # define training loop
    test_loss_min = np.Inf
    train_loss_min = np.Inf

    train_loss_list = list()
    test_loss_list = list()
    
    model = model.to(device)

    start = time()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0
        total_correct_train = 0
        total_correct_test = 0
        total_train = 0
        total_test = 0
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

        # validate model
        # model.eval()
        # for data, target in test_loader:
        #     data, target = data.to(device), target.to(device)
        #     with torch.no_grad():
        #         output = model(data)
        #         loss = criterion(output, target)
        #         test_loss += loss.item()

        #         total_test += target.shape[0]

        if scheduler != None:
            # update scheduler
            scheduler.step()

        # compute average loss
        train_loss /= total_train
        #test_loss /= total_test

        # save data
        train_loss_list.append(train_loss)
        #test_loss_list.append(test_loss)


        # display stats with test
        #print('Epoch: {}/{} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, n_epochs, train_loss, test_loss))
        
        # display stats
        print('Epoch: {}/{} \tTrain Loss: {:.6f}'.format(epoch, n_epochs, train_loss))

        # save best model
        # if test_loss <= test_loss_min:
        #     print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(test_loss_min, test_loss))

        #     if not os.path.isdir('best_model'):
        #         os.mkdir('best_model')

        #     torch.save(model.state_dict(), './best_model/model1.pt')
        #     test_loss_min = test_loss
        
        # save best model
        if train_loss <= train_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(train_loss_min, train_loss))

            if not os.path.isdir('best_model'):
                os.mkdir('best_model')

            torch.save(model.state_dict(), './best_model/model1.pt')
            train_loss_min = train_loss
    
    end = time()

    print('Time elapsed: {} hours'.format((end - start) / 3600.0))
    return model, train_loss_list, test_loss_list