import json
from neuralop.models import TFNO
import torch

from train.trainer import PhysicsInformedLoss, Trainer
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    with open("config.json") as config_data:
        config = json.load(config_data)
    
    model = TFNO(
        n_modes = config['model']['n_modes'],
        in_channels = config['model']['in_channels'],
        out_channels = config['model']['out_channels'],
        hidden_channels = config['model']['hidden_channels'],
        n_layers = config['model']['n_layers']
    )


    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhysicsInformedLoss(config['training']['loss_weights'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )

    trainer.train()
