import torch
from torch.utils.data import Dataset
import os
import numpy as np


class WildFireData(Dataset):
    def __init__(self, data_dir):
        '''
            initialize the dataset
        '''

    def __len__(self):
        return self.data.shape[0] 

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1].view(1)
        return features, label