import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms as T

class RunningDataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.img_stack = np.zeros((batch_size,3,224,224), dtype="float32")
        self.data_stack = np.zeros((batch_size), dtype="float32")

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.img_stack[idx], np.array([self.data_stack[idx]])
      
    def __setitem__(self, idx, value):
        self.img_stack[idx], self.data_stack[idx] = value


def autonomous_preprocess(img,img_dim = (224, 224)):
    to_tensor = T.ToTensor()
    # img = cv2.GaussianBlur(img,  (3, 3), 0) 
    img = cv2.resize(img,img_dim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # img = img/255
    img = to_tensor(img)
    img = torch.unsqueeze(img,dim=0)
    return img
