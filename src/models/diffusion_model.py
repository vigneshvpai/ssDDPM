import torch
import diffusers
from torchvision import transforms
import lightning as L
import nibabel as nib
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Enable Tensor Core optimization for A100 GPU
torch.set_float32_matmul_precision("high")


class DiffusionMRIDataset(Dataset):
    """Custom dataset for loading diffusion MRI NIfTI files"""

    def __init__(self, data_dir, transform=None, target_size=(64, 64)):
        """
        Args:
            data_dir (str): Path to the data directory containing earthworm folders
            transform (callable, optional): Optional transform to be applied
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size

        # Find all NIfTI files
        self.nii_files = []
        for earthworm_dir in self.data_dir.iterdir():
            if earthworm_dir.is_dir() and earthworm_dir.name.startswith("earthworm"):
                for file in earthworm_dir.glob("*.nii.gz"):
                    self.nii_files.append(file)

        print(
            f"Found {len(self.nii_files)} NIfTI files in {len(list(self.data_dir.glob('earthworm*')))} earthworm directories"
        )

    def __len__(self):
        return len(self.nii_files)

    def __getitem__(self, idx):
        nii_file = self.nii_files[idx]

        # Load NIfTI data
        img = nib.load(str(nii_file))
        data = img.get_fdata()

        # Convert to torch tensor and normalize
        data = torch.from_numpy(data).float()

        # Normalize to [0, 1] range
        if data.max() > 0:
            data = data / data.max()

        # Take a middle slice for 2D processing (you can modify this for 3D)
        # For now, we'll use the middle slice of the first diffusion direction
        if len(data.shape) == 4:  # (x, y, z, diffusion_directions)
            middle_z = data.shape[2] // 2
            data = data[
                :, :, middle_z, 0
            ]  # Take middle z-slice, first diffusion direction
        elif len(data.shape) == 3:  # (x, y, diffusion_directions)
            data = data[:, :, 0]  # Take first diffusion direction

        # Ensure 2D
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")

        # Resize to target size
        data = torch.nn.functional.interpolate(
            data.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # Remove batch and channel dims

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return {"images": data.unsqueeze(0)}  # Add channel dimension


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
