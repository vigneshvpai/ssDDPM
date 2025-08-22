from diffusers import UNet2DModel, DDPMScheduler
import lightning as L
import torch
import torch.nn as nn
from src.config.config import Config
import torchvision.utils as vutils


class SSDDPM(L.LightningModule):
    def __init__(self, in_channels, out_channels, lr=1e-4):
        super().__init__()
        self.model = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

        self.scheduler = DDPMScheduler(**Config.SCHEDULER_CONFIG)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **Config.OPTIMIZER_CONFIG)

        # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        images = batch
        noise = torch.randn_like(images)
        steps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        self.log_images(images, noisy_images, residual, "train")
        return loss

    def log_images(self, clean_images, noisy_images, predicted_noise, prefix):
        """Log images to TensorBoard"""
        # clean_images shape: [batch, bvalues*slices, H, W]
        # For visualization, select the first sample in the batch and the first 3 channels (bvalue/slice combinations)
        # Adjust indices as needed for your use case

        # Select the first sample in the batch
        clean_vis = clean_images[0, 10:13, :, :]
        noisy_vis = noisy_images[0, 10:13, :, :]
        noise_vis = predicted_noise[0, 10:13, :, :]

        # Create a grid of images
        grid_clean = vutils.make_grid(clean_vis, nrow=1, normalize=True)
        grid_noisy = vutils.make_grid(noisy_vis, nrow=1, normalize=True)
        grid_noise = vutils.make_grid(noise_vis, nrow=1, normalize=True)

        # Log to TensorBoard
        self.logger.experiment.add_image(
            f"{prefix}_clean_images", grid_clean, self.current_epoch
        )
        self.logger.experiment.add_image(
            f"{prefix}_noisy_images", grid_noisy, self.current_epoch
        )
        self.logger.experiment.add_image(
            f"{prefix}_predicted_noise", grid_noise, self.current_epoch
        )
