import torch
import diffusers
from torchvision import transforms
import lightning as L
import os
from torch.utils.data import DataLoader
from src.data.dataset import DiffusionMRIDataset

# Enable Tensor Core optimization for A100 GPU
torch.set_float32_matmul_precision("high")


class DiffusionModel(L.LightningModule):
    def __init__(self, data_dir="/home/vault/mfdp/mfdp118h/data", target_size=(64, 64)):
        super().__init__()
        self.data_dir = data_dir
        self.target_size = target_size

        # Initialize a simple UNet model for 2D images
        self.model = diffusers.models.UNet2DModel(
            sample_size=target_size[0], in_channels=1, out_channels=1
        )
        # Enable gradient checkpointing to save memory during backpropagation
        self.model.enable_gradient_checkpointing()
        # Use the default DDPM scheduler for noise scheduling
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        # Get the images from the batch
        images = batch["images"]
        # Add channel dimension if not present
        if len(images.shape) == 3:  # (batch, height, width)
            images = images.unsqueeze(1)  # (batch, channel, height, width)

        # Generate random Gaussian noise with the same shape as images
        noise = torch.randn_like(images)
        # Randomly select a timestep for each image in the batch
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,  # total number of timesteps
            (images.size(0),),  # one timestep per image
            device=self.device,
        )
        # Add noise to the images according to the selected timesteps
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # Predict the noise residual using the model
        residual = self.model(noisy_images, steps).sample
        # Compute mean squared error loss between predicted and true noise
        loss = torch.nn.functional.mse_loss(residual, noise)
        # Log the training loss for monitoring
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer for training
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # Use a step learning rate scheduler that decays every epoch
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        """Create dataloader for training"""
        dataset = DiffusionMRIDataset(
            data_dir=self.data_dir, target_size=self.target_size
        )
        return DataLoader(
            dataset,
            batch_size=8,  # Adjust based on your GPU memory
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
