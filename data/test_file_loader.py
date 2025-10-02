import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch


class TestFileLoader:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_data_path = config['data']['test_path']
        self.files = []
        self.files.extend(list(Path(self.test_data_path).glob('**/input/*.npz')))
        self.lenght = len(self.files)

    def get_input(self, file_idx):
        file_name = os.path.basename(self.files[file_idx])
        input_file = os.path.join(self.test_data_path, 'input', file_name)

        with np.load(input_file, mmap_mode='r') as data:
            input = torch.from_numpy(data['b0'].astype(np.float32))

        return input
    
    def get_inputs_list(self):
        input_dict_list = {"input": []}
        for file_idx in range(self.lenght):
            input = self.get_input(file_idx)
            input_dict_list['input'].append(input)
        return input_dict_list
    
    def get_outputs_list(self, model, input_dict_list):
        output_dict_list = {"output": []}

        with torch.no_grad():
            for input in input_dict_list['input']:
                print("Processing new input")
                input = input.to(self.device)
                output = model(input)
                output_dict_list['output'].append(output.cpu())
        return output_dict_list
    
    def get_label(self, file_idx):
        file_name = os.path.basename(self.files[file_idx])
        noaa_ar = file_name[:5]
        
        label_file = os.path.join(self.test_data_path, 'label', file_name)

        with np.load(label_file, mmap_mode='r') as data:
            b = torch.from_numpy(data['b'].astype(np.float32))
            x = data['x'].astype(np.float32)
            y = data['y'].astype(np.float32)

        b = b[:, :-1, :-1, :-1]

        return noaa_ar, b, x, y
    
    def get_labels_list(self):
        label_dict_list = {"noaa_ar": [], "b": [], "x": [], "y": []}
        for file_idx in range(self.lenght):
            noaa_ar, b, x, y = self.get_label(file_idx)
            label_dict_list['noaa_ar'].append(noaa_ar)
            label_dict_list['b'].append(b)
            label_dict_list['x'].append(x)
            label_dict_list['y'].append(y)
        return label_dict_list
    
    def plot_ground_truth(self, noaa_ar, bz0, x, y):
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, bz0.T, cmap='gray', vmin=-2000, vmax=2000)
        ax.set_xlabel('x [Mm]')
        ax.set_ylabel('y [Mm]')
        ax.set_aspect('equal')
        ax.set_title(f"Bz at z=0 of NOAA {noaa_ar}")
        plt.tight_layout()
        plt.show()      
