from preprocess import img_preprocess

import torch
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import numpy as np



class DrivingImageDataset(Dataset):
    def __init__(self, image_paths, steering_ang, transform=None):
        self.img_paths = image_paths
        self.steering_angs = steering_ang
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = mpimg.imread(img_path)
        img = img_preprocess(img)
        img = img.astype(np.uint8)
        steering = self.steering_angs[index]

        if self.transform is not None:
            img = self.transform(img)
            # img = img.to(dtype=torch.long)
            steering = torch.tensor(steering)
            steering = torch.tensor([steering.to(dtype=torch.float32)])

        # print('steering: ',steering.shape)
        return img, steering