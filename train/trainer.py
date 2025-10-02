import os
import gc
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError

from data.isee_dataset import ISEEDataset

def divergence(bx, by, bz):
    """
    Calculates the divergence of a 3D vector field B = (bx, by, bz).
    Assumes a grid spacing of 1.0 in all directions.

    Args:
        bx (torch.Tensor): The x-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).
        by (torch.Tensor): The y-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).
        bz (torch.Tensor): The z-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).

    Returns:
        torch.Tensor: The divergence of the field.
                      Shape: (batch_size, Nz, Ny, Nx).
    """
    grad_bx_dx = torch.gradient(bx, dim=3)[0]
    grad_by_dy = torch.gradient(by, dim=2)[0]
    grad_bz_dz = torch.gradient(bz, dim=1)[0]
    return grad_bx_dx + grad_by_dy + grad_bz_dz

def curl(bx, by, bz):
    """
    Calculates the curl of a 3D vector field B = (bx, by, bz).
    Assumes a grid spacing of 1.0 in all directions.

    Args:
        bx (torch.Tensor): The x-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).
        by (torch.Tensor): The y-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).
        bz (torch.Tensor): The z-component of the vector field.
                           Shape: (batch_size, Nz, Ny, Nx).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The components (jx, jy, jz) of the curl.
                                                          Each has shape (batch_size, Nz, Ny, Nx).
    """
    dbz_dy = torch.gradient(bz, dim=2)[0]
    dby_dz = torch.gradient(by, dim=1)[0]

    dbx_dz = torch.gradient(bx, dim=1)[0]
    dbz_dx = torch.gradient(bz, dim=3)[0]

    dby_dx = torch.gradient(by, dim=3)[0]
    dbx_dy = torch.gradient(bx, dim=2)[0]

    jx = dbz_dy - dby_dz
    jy = dbx_dz - dbz_dx
    jz = dby_dx - dbx_dy
    return jx, jy, jz

class PhysicsInformedLoss(torch.nn.Module):
    """
    A custom loss function that incorporates physical constraints.
    The total loss is a weighted sum of:
    - MSE: Standard Mean Squared Error between prediction and label.
    - Bottom Boundary Condition: MSE loss on the bottom (z=0) boundary.
    - Divergence-Free Loss: Penalizes non-zero divergence of the magnetic field (∇ ⋅ B).
    - Force-Free Loss: Penalizes the Lorentz force (J x B), encouraging a force-free field.
    """
    def __init__(self, loss_weights: dict):
        """
        Initializes the loss function.

        Args:
            loss_weights (dict): A dictionary with weights for each loss component.
        """
        super().__init__()
        self.weights = loss_weights
        self.mse_loss = MeanSquaredError()

    def forward(self, outputs, labels):
        """
        Calculates the total physics-informed loss.

        Args:
            outputs (torch.Tensor): The model's predictions.
                                    Shape: (batch_size, 3, Nz, Ny, Nx).
            labels (torch.Tensor): The ground truth labels.
                                   Shape: (batch_size, 3, Nz, Ny, Nx).

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing the total loss tensor
                                       and a dictionary of the individual loss components.
        """
        self.mse_loss.to(outputs.device)

        loss_mse = self.mse_loss(outputs.flatten(), labels.flatten())

        b_pred = torch.permute(outputs, (0, 2, 3, 4, 1))
        b_true = torch.permute(labels, (0, 2, 3, 4, 1))

        z_dim_size = b_pred.shape[1]
        divisor = (1.0 / torch.arange(1, z_dim_size + 1, device=b_pred.device)).reshape(1, -1, 1, 1, 1)
        b_pred_denorm = b_pred * divisor
        b_true_denorm = b_true * divisor

        loss_bc_bottom = self.mse_loss(
            b_pred_denorm[:, 0, :, :, :].flatten(),
            b_true_denorm[:, 0, :, :, :].flatten()
        )

        bx, by, bz = b_pred_denorm.unbind(dim=-1)

        div_b = divergence(bx, by, bz)
        loss_div = torch.mean(div_b**2)

        jx, jy, jz = curl(bx, by, bz)
        j = torch.stack([jx, jy, jz], dim=-1)

        jxb = torch.cross(j, b_pred_denorm, dim=-1)
        lorentz_force_sq = (jxb**2).sum(-1)
        magnetic_energy_sq = (b_pred_denorm**2).sum(-1)

        loss_ff = torch.mean(lorentz_force_sq / (magnetic_energy_sq + 1e-8))

        total_loss = (
            self.weights['w_mse'] * loss_mse +
            self.weights['w_bc_bottom'] * loss_bc_bottom +
            self.weights['w_ff'] * loss_ff +
            self.weights['w_div'] * loss_div
        )

        loss_dict = {
            'mse': loss_mse, 'bc_bottom': loss_bc_bottom,
            'ff': loss_ff, 'div': loss_div,
        }
        return total_loss, loss_dict

class Trainer:
    """
    Handles the model training and validation process.
    """
    def __init__(self, model, optimizer, loss_fn, config):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            optimizer: The optimization algorithm (e.g., Adam).
            loss_fn: The loss function to use.
            config (dict): A configuration dictionary containing training and data parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def _create_dataloaders(self):
        """Creates and returns the data loaders for training and validation."""
        train_dataset = ISEEDataset(
            data_path=self.config['data']['dataset_path'], b_ground=self.config['data']['b_ground']
        )
        val_dataset = ISEEDataset(
            data_path=self.config['data']['val_path'], b_ground=self.config['data']['b_ground']
        )
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, pin_memory=True
        )
        return train_loader, val_loader

    def _forward(self, batch):
        """
        Performs a forward pass and loss calculation, shared between training and validation.

        Args:
            batch (dict): A batch of data from the DataLoader.

        Returns:
            Tuple[torch.Tensor, dict]: The total loss and a dictionary of individual losses.
        """
        inputs = batch['input'].to(self.device).permute(0, 4, 1, 2, 3)
        labels = batch['label'].to(self.device).permute(0, 4, 1, 2, 3)

        outputs = self.model(inputs)

        return self.loss_fn(outputs, labels)

    def _train_epoch(self, epoch):
        """
        Runs a single training epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", ncols=100)
        for batch in progress_bar:
            loss, loss_dict = self._forward(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4e}")

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} | Average Training Loss: {avg_loss:.6f}")

    def _validate_epoch(self, epoch):
        """
        Runs a single validation epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", ncols=100, leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                loss, loss_dict = self._forward(batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch} | Validation Loss: {avg_loss:.6f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"✨ New best model found with validation loss: {self.best_val_loss:.6f}")
            self._save_checkpoint(epoch, "best_model.pt")

    def _save_checkpoint(self, epoch, filename):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): The current epoch number.
            filename (str): The name of the checkpoint file.
        """
        checkpoint_dir = self.config['model_save_path']
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, os.path.join(checkpoint_dir, filename))

    def train(self):
        """Starts the main training loop."""
        print(f"Starting training on {self.device}...")
        for epoch in range(self.start_epoch, self.config['training']['n_epochs']):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch, "last_model.pt")

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        print("Training finished.")