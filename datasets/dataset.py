import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

class MRSDataset(Dataset):
    """
    Output data shape: (512) np.float32
    487-999 corresponds to 0.2ppm-4.5ppm
    """
    def __init__(self, root, split, data_type='low_lipid'):
        if data_type == 'all':
            self.data_dir = os.path.join(root,split)
            self.dataset = []
            # Iterate over each directory within the split directory
            for subdir in os.listdir(self.data_dir):
                subdir_path = os.path.join(self.data_dir, subdir)
                if os.path.isdir(subdir_path):  # Check if it's a directory
                    # Load all files within the subdirectory
                    for fn in os.listdir(subdir_path):
                        if fn.endswith('.tsv'): 
                            self.dataset.append({"fn": os.path.join(subdir_path, fn)})
        else:
            self.data_dir = os.path.join(root,split,data_type)
            self.dataset = []
            if os.path.isdir(self.data_dir):
                for fn in os.listdir(self.data_dir):
                    if fn.endswith('.tsv'):
                        self.dataset.append({"fn": os.path.join(self.data_dir, fn)})

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        fn = self.dataset[idx]['fn']
        data = pd.read_csv(fn, sep='\t')
        FD_Re = data['FD_Re'].values[487:999].astype(np.float32)
        return FD_Re