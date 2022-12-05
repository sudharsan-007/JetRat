import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
import cv2
import torchvision.transforms as T


class SimulatorDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, imgs_folder, img_dim = (320, 240), transform = None):
        self.columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse','speed']
        self.df = pd.read_csv(csv_path, names = self.columns)
        self.df = self.df.drop(['left','right'],axis=1)
        self.df['center'] = self.df["center"].apply(self._path_leaf)
        self.imgs_folder = imgs_folder
        self.transform = transform 
        self.img_dim = img_dim
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self,index):
        label = np.array([self.idx_key_df(index,"steering"),self.idx_key_df(index,"throttle")],dtype="float32") 
        imgs_path = os.path.join(self.imgs_folder, self.idx_key_df(index,"center")) 
        image = cv2.imread(imgs_path) 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG to RGB for training 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BRG to HSV for training 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # BRG to YUV for training 
        image = cv2.resize(image, self.img_dim)
        image = self.to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)
        return np.array(image), label #.reshape(-1, 2) 

    def get_rgb(self,index):
        label = np.array([self.idx_key_df(index,"steering"),self.idx_key_df(index,"throttle")])
        imgs_path = os.path.join(self.imgs_folder, self.idx_key_df(index,"center"))
        image = cv2.imread(imgs_path) # OpenCV reads images in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG to RGB for training 
        image = cv2.resize(image, self.img_dim)
        image = self.to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)
        return image.permute(1, 2, 0), label
        
    def _path_leaf(self, path):
        path = path.replace("\\","/") 
        head, tail = os.path.split(path) 
        return tail
    
    def index_data(self,idx):
        print(self.df.iloc[idx])
    
    def idx_key_df(self,idx,key): # checked with iloc also this is faster
        return self.df.loc[idx,key]
    
    def show_data(self):
        print(self.df.head()) # Or do df.head() when using jupyter-notebook
    
    def plot_hist(self, num_bins=25, samples_per_bin= 1500):
        hist, bins = np.histogram(self.df['steering'], num_bins)
        center = (bins[:-1]+bins[1:]) * 0.5
        plt.bar(center, hist, width = 0.05)
        plt.plot((np.min(self.df['steering']), np.max(self.df['steering'])), (samples_per_bin,samples_per_bin))
        
    def balance_data(self, num_bins=25, samples_per_bin= 1500):
        print('total data', len(self.df))
        hist, bins = np.histogram(self.df['steering'], num_bins)
        remove_list = []
        for j in range(num_bins):
            list_ = []
            for i in range(len(self.df['steering'])):
                if self.df['steering'][i] >= bins[j] and self.df['steering'][i] <= bins[j+1]:
                    list_.append(i)
            # list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)
            
        print('removed', len(remove_list))
        self.df.drop(self.df.index[remove_list], inplace = True)
        print('remaining', len(self.df))
        
    def set_gain(self, Gain):
        self.df["steering"] = self.df["steering"] * Gain
        self.df["steering"] = self.df["steering"].clip(lower=-1.0, upper=1.0)
     
    def SaveProcessedData(self):
        self.df.to_csv("simulator_data/clean_sim_data.csv") 

class SimLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, imgs_folder, img_dim = (320, 240), transform = None):
        self.df = pd.read_csv(csv_path)
        self.imgs_folder = imgs_folder
        self.transform = transform 
        self.img_dim = img_dim
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self,index):
        label = np.array([self.idx_key_df(index,"steering"),self.idx_key_df(index,"throttle")],dtype="float64") 
        imgs_path = os.path.join(self.imgs_folder, self.idx_key_df(index,"center")) 
        image = cv2.imread(imgs_path) 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG to RGB for training 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BRG to HSV for training 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # BRG to YUV for training 
        image = cv2.resize(image, self.img_dim) 
        image = self.to_tensor(image) 
        if self.transform is not None:
            image = self.transform(image) 
        return image, label.reshape(-1, 2) 



class JetcarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, imgs_folder, img_dim = (320, 240), transform = None):
        # self.columns = ['img_epoch', 'throttle', 'steering']
        self.df = pd.read_csv(csv_path)
        self.imgs_folder = imgs_folder
        self.transform = transform 
        self.img_dim = img_dim
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self,index):
        label = np.array([self.idx_key_df(index,"Steering"),self.idx_key_df(index,"Throttle")],dtype="float32")
        imgs_path = os.path.join(self.imgs_folder, self.idx_key_df(index,"Epoch_time"))
        image = cv2.imread(imgs_path+".png")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG to RGB for training 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BRG to HSV for training 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # BRG to YUV for training 
        image = cv2.resize(image, self.img_dim)
        image = self.to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)
        return np.array(image), label
    
    def get_rgb(self,index):
        label = np.array([self.idx_key_df(index,"Steering"),self.idx_key_df(index,"Throttle")],dtype="float32")
        imgs_path = os.path.join(self.imgs_folder, self.idx_key_df(index,"Epoch_time"))
        image = cv2.imread(imgs_path+".png") # OpenCV reads images in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG to RGB for training 
        image = cv2.resize(image, self.img_dim)
        image = self.to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)
        return image.permute(1, 2, 0), label
    
    def index_data(self,idx):
        print(self.df.iloc[idx])
    
    def idx_key_df(self,idx,key): # checked with iloc also this is faster
        return self.df.loc[idx,key]
    
    def show_data(self):
        print(self.df.head()) # Or do df.head() when using jupyter-notebook
    
    def plot_hist(self, num_bins=7, samples_per_bin= 35):
        hist, bins = np.histogram(self.df['Steering'], num_bins)
        center = (bins[:-1]+bins[1:]) * 0.5
        plt.bar(center, hist, width = 0.05)
        plt.plot((np.min(self.df['Steering']), np.max(self.df['Steering'])), (samples_per_bin,samples_per_bin))
        
    def balance_data(self, num_bins=7, samples_per_bin= 35):
        print('total data', len(self.df))
        hist, bins = np.histogram(self.df['Steering'], num_bins)
        remove_list = []
        for j in range(num_bins):
            list_ = []
            for i in range(len(self.df['Steering'])):
                if self.df['Steering'][i] >= bins[j] and self.df['Steering'][i] <= bins[j+1]:
                    list_.append(i)
            # list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)
            
        print('removed', len(remove_list))
        self.df.drop(self.df.index[remove_list], inplace = True)
        print('remaining', len(self.df))
        
    def set_gain(self, Gain):
        self.df["Steering"] = self.df["Steering"] * Gain
        self.df["Steering"] = self.df["Steering"].clip(lower=-1.0, upper=1.0)




