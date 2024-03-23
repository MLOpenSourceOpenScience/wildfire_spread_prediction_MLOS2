import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from tqdm import tqdm


class WildFireData(Dataset):
    def __init__(self, data_dir, load_dir, save_dir, patch_size, step_size):
        '''
            initialize the dataset
        '''

        if load_dir is not None:
            load_dict = torch.load(load_dir)
            self.features = load_dict["features"]
            self.labels = load_dict["labels"]
            self.mean = load_dict["mean"]
            self.std = load_dict["std"]
            self.mean = self.mean.view(-1, 1, 1)
            self.std = self.std.view(-1, 1, 1)
            return

        maps = [d for d in sorted(os.listdir(data_dir))]
        self.features = []
        self.labels = []
        current = None
        next = None
        with open(os.path.join(data_dir, maps[0]), 'rb') as file:
            current = pickle.load(file)
        for map in tqdm(maps[1:], total=(len(maps)-1)):
            with open(os.path.join(data_dir, map), 'rb') as file:
                next = pickle.load(file)
            if (current["time"][0,0,0]+1) == (next["time"][0,0,0]):
                features = np.concatenate((current["data"], current["fire"]), axis=0)
                labels = next["fire"]
                features = torch.tensor(features)
                labels = torch.tensor(labels)
                features = features.unfold(1, size=patch_size, step=step_size)
                features = features.unfold(2, size=patch_size, step=step_size)
                features = torch.einsum('chwjk->hwcjk', features)
                features = features.reshape(-1, features.shape[2], patch_size, patch_size)

                labels = labels.unfold(1, size=patch_size, step=step_size)
                labels = labels.unfold(2, size=patch_size, step=step_size)
                labels = torch.einsum('chwjk->hwcjk', labels)
                labels = labels.reshape(-1, labels.shape[2], patch_size, patch_size)

                fires = features[:,-1]
                fires = fires.view(-1, patch_size*patch_size)

                features = features[torch.sum(fires, dim=1)>10]
                labels = labels[torch.sum(fires, dim=1)>10]
                self.features.extend([f for f in features])
                self.labels.extend([l for l in labels])
                
            current = next
        # self.features = torch.concat(self.features, dim=0)
        # self.labels = torch.concat(self.labels, dim=0)

        features = torch.stack(self.features)
        self.mean = torch.mean(features, dim=(-1, -2)).mean(dim=0)
        self.std = torch.std(features, dim=(-1, -2)).mean(dim=0)

        save_dict = {
            "features": self.features,
            "labels": self.labels,
            "mean": self.mean,
            "std": self.std
        }
        torch.save(save_dict, save_dir)


    def __len__(self):
        # return self.features.shape[0] 
        return len(self.features)

    def __getitem__(self, idx):
        feature = (self.features[idx] - self.mean) / self.std
        # feature = self.features[idx]
        return feature.type(torch.float32), self.labels[idx].flatten().type(torch.float32)