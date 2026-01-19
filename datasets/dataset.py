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
    def __init__(self, root, split, data_type='metab_lipid'):
        self.data_dir = os.path.join(root,split,data_type)
        self.files = sorted(os.listdir(self.data_dir))
        self.dataset = [{"fn": fn} for fn in self.files if fn.endswith('.tsv')]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        fn = self.dataset[idx]['fn']
        data = pd.read_csv(os.path.join(self.data_dir, fn), sep='\t')
        FD_Re = data['FD_Re'].values[487:999].astype(np.float32)
        return FD_Re