import numpy as np
import torch
from neuralop.models import TFNO

class Model:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TFNO(
            n_modes = config['model']['n_modes'],
            in_channels = config['model']['in_channels'],
            out_channels = config['model']['out_channels'],
            hidden_channels = config['model']['hidden_channels'],
            n_layers = config['model']['n_layers']
        ).to(self.device)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def __call__(self, input):
        input = input[np.newaxis, :, :, :, :] 
        #input -> [1, 3, 512, 256, 1]
        input = input[:, :, :-1, :-1, :]
        #input -> [1, 1, 256, 512, 3]
        input = torch.permute(input, (0, 4, 3, 2, 1))
        model_input = input.to(self.device)
        
        #model_output -> [1, 256, 256, 512, 3]
        model_output = self.model(model_input)
        #model_output -> [3, 512, 256, 256]
        model_output = torch.permute(model_output.detach().cpu(), (0, 4, 3, 2, 1))[0]
        nz_size = model_output.shape[3]
        for i in range(nz_size):
            model_output[:, :, :, i] = model_output[:, :, :, i] / (i + 1)
            
        model_output = model_output
        return model_output



