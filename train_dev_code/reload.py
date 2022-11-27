import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

import os
from utils.model import *
from utils.dataset import DrivingImageDataset
from utils.process_bar import progress_bar
#---------
def load_img_steering(datadir, data):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        # left image append
        image_path.append(os.path.join(datadir,left.strip()))
        steering.append(float(indexed_data[3])+0.15)
        # right image append
        image_path.append(os.path.join(datadir,right.strip()))
        steering.append(float(indexed_data[3])-0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

def load_data(datadir):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
    pd.set_option('display.max_colwidth', None)
    data.head()
    return data
#---------

def train_reload(trainloader, PATH='./checkpoint/ckpt.pth', net=nvidia_model()):
    '''
    Load the model from 'PATH', use data from 'trainloader' to further train
    the model 'net'
    Return: net
    '''
    device = torch.device('cpu')
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    checkpoint = torch.load(PATH, map_location=device)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss()

    net.train()
    train_loss = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        verboseFlag = False
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # torch.cuda.synchronize()

        train_loss += loss.item()
        total += targets.size(0)

        if not verboseFlag:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.5f'
                    % (train_loss/(batch_idx+1)))
            
    return net
    
def test_reload(testloader, best_loss, PATH='./checkpoint/ckpt.pth', net=nvidia_model()):
    '''
    Load the model from 'PATH', use data from 'testloader' to test the loss on
    'net'. If the loss is lesser than the 'best_loss', save the checkpoint
    Return: net
    '''
    verboseFlag = False
    
    device = torch.device('cpu')
    PATH = './checkpoint/ckpt.pth'
    net = nvidia_model()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    checkpoint = torch.load(PATH, map_location=device)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss()
    
    net.eval()
    test_loss = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)

        if not verboseFlag:
            progress_bar(batch_idx, len(testloader), 'Loss: %.5f'
                    % (test_loss/(batch_idx+1)))

    # Save checkpoint.
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = loss
    return best_loss

def gen_Dataset(Dir, batch):
    '''
    Generate the dataset object based on directories (can be modified to be
    image array or others) 
    '''
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor()
    ])
    data = load_data(Dir)
    image_paths, steerings = load_img_steering(Dir + '/IMG', data)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2)
    print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))
    Q = math.floor(len(X_train)/batch)
    trainset = DrivingImageDataset(X_train, y_train, transform_train)
    testset = DrivingImageDataset(X_valid, y_valid, transform_valid)
    return trainset, testset