import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomCSVDataset(Dataset):

    def __init__(self, root: str, feature_columns: list[str], 
                 label_columns: list[str], is_regression_task: bool):

        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.root = root

        self.df = pd.read_csv(root)
        self.feature_columns_data = self.df[feature_columns].values
        self.label_columns_data = self.df[label_columns].values

        self.is_regression_task = is_regression_task
        self.is_multi_label_classification = not is_regression_task and len(self.label_columns) > 1

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        features = torch.from_numpy(self.feature_columns_data[index]).float()
        labels = torch.from_numpy(self.label_columns_data[index])

        if self.is_regression_task:
            # For any regression task (single or multi-output), labels should be float
            labels = labels.float()
        elif self.is_multi_label_classification:
            # For multi-label classification, labels (typically 0s and 1s) should be float
            labels = labels.float()
        else:
            # This is a single-label classification task
            # Ensure labels are squeezed and cast to long for class IDs
            if labels.dim() > 0 and labels.size(0) == 1:
                labels = labels.squeeze(0).long() # Squeeze and cast to long for class IDs
            else:
                labels = labels.long() # Already a scalar or correctly shaped 1D for class ID

        return features, labels


        