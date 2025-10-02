import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os

class ISEEDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing 3D physics simulation data.

    This class handles loading input (.npz files), performing normalization,
    and removing periodic boundaries.

    Args:
        data_paths (str): Path to dataset directories.
        b_ground (float): Normalization constant for the magnetic field.
    """
    def __init__(self, data_path, b_ground):
        """
        Initializes the dataset by finding all input files.

        Args:
            data_path (str): The root directory path for the data.
            b_ground (float): The normalization factor for the magnetic field data.
        """
        self.files = []
        self.files.extend(list(Path(data_path).glob('**/input/*.npz')))
        self.data_path = data_path
        self.files = sorted(self.files)
        self.length = len(self.files)
        self.b_ground = b_ground

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single data sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed 'input', 'label',
                  and 'input_name'.
        """
        file_name = os.path.basename(self.files[idx])
        input_file = os.path.join(self.data_path, 'input', file_name)
        label_file = os.path.join(self.data_path, 'label', file_name)

        with np.load(input_file, mmap_mode='r') as data:
            inputs = torch.from_numpy(data['b0'].astype(np.float32))

        with np.load(label_file, mmap_mode='r') as data:
            labels = torch.from_numpy(data['b'].astype(np.float32))

        inputs = inputs[:, :-1, :-1, :]
        labels = labels[:, :-1, :-1, :-1]

        inputs = inputs / self.b_ground

        nz_size = labels.shape[3]
        for i in range(nz_size):
            labels[:, :, :, i] = (labels[:, :, :, i] * (i+1)) / (self.b_ground)

        sample = {
            'input': inputs, 'label': labels,
        }

        return sample