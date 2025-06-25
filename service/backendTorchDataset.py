from config import logging
from typedefs import *

import torch
from torch import nn
from torch.utils.data import Dataset

import pandas as pd

class TorchCustomCSVDataset(Dataset):

    def __init__(self, path_to_csv: str, feature_columns: list[str], label_columns: list[str]):

        self.df = pd.read_csv(path_to_csv)
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        # convert them to numpy array
        self.feature_data = self.df[feature_columns].values
        self.label_data = self.df[label_columns].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        features = self.feature_data[index]
        labels = self.label_data[index]

        features_tensor = torch.tensor(features, dtype=torch.float32)

        # single label task like classification, convert them to long
        if len(self.label_columns) == 1: 
            labels_tensor = torch.tensor(labels, dtype=torch.long).squeeze()
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return features_tensor, labels_tensor



    


