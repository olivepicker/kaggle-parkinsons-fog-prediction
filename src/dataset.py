import numpy as np
import torch
import cv2
import librosa as lib
import torch.nn.functional as F
import os
import tqdm
import albumentations

from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset



####################################################################################################

class CNN2dDataset(Dataset):
    def __init__(self, df, npy, time_to, model_type='past', augment=None):
        
        self.df = df
        self.npy = npy
        self.augment = augment
        self.model_type = model_type
        self.time_to = time_to
        self.length = len(self.df)

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        d = self.df.iloc[i]
        id = d['id']
        time = d['Time']
        data = self.npy[id]
        time_to = self.time_to

        shape = int(time_to **(1/2))
        if self.model_type == 'past' :
            if time >= time_to : 
                out = data[time-(time_to-1):time+1, :].T
                out = np.asarray(out).reshape(3, shape, -1).astype(float)
            else :
                arr = np.zeros((3, time_to))
                out = data[:time, :].T
                arr[:,time_to-time:] = out
                out = arr.reshape(3, shape, -1).astype(float)
                
        elif self.model_type == 'current' :
            start = 0
            end = 0
            status = 0
            if time - (time_to // 2) < 0 :
                start = 0
                status = 1
            else :
                start = time - (time_to // 2)

            if time + (time_to // 2) > data.shape[0] :
                end = data.shape[0]
                status = 2
            else :
                end = time + (time_to // 2)
                
            arr = np.zeros((time_to, 3))
            out = data[start : end, : ]
            out_len = out.shape[0]
            if status == 0 :
                arr[:,:] = out
            elif status == 1 :
                arr[time_to-out_len:, :] = out
            else :
                arr[:out_len, :] = out
            out = arr.reshape(3, -1)

        elif self.model_type == 'future' :
            start = time
            end = 0
            
            if time + time_to > data.shape[0] :
                end = data.shape[0]
            else :
                end = time + time_to
            arr = np.zeros((3,time_to))
            
            out = data[start : end, : ].T
            arr[:,:end-start] = out
            out = arr.reshape(3, shape, -1).astype(float)
                
        label = d[['StartHesitation','Turn','Walking']].values.astype(float)
        
        if self.augment is not None :
            out = self.augment(image=out.transpose(1,2,0))['image']
            out = out.transpose(2,0,1)

        ret = {
            'signals' : torch.from_numpy(out).float(),
            'label'   : torch.from_numpy(label),
        }

        return ret
    
def train_augment():
    aug = albumentations.Compose([
        albumentations.VerticalFlip(p=0.1),
        albumentations.HorizontalFlip(p=0.1),
        albumentations.RandomGridShuffle(grid=(3, 3), p= 0.1),
        albumentations.OneOf([
            albumentations.RingingOvershoot((1,2), cutoff = (0.7853981633974483, 1.207963267948966),p=0.1),
            albumentations.Sharpen(alpha=(0.01, 0.05),lightness=(0.01,0.05),p=0.1),
        ], p=0.5),  
        albumentations.OneOf([
            albumentations.CoarseDropout(max_holes = 4, max_height=9, max_width = 9, p=0.1),
            albumentations.ChannelDropout(channel_drop_range=(1, 1), fill_value = 0., p=0.1),
            albumentations.PixelDropout(drop_value=0.,per_channel=True, p = 0.1),
        ], p=0.5)

    ])
    
    return aug
    
def valid_augment():
    aug = None
    return aug
