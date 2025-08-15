import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L
from typing import Optional, Dict, Any
import numpy as np

from src.models.custom_scheduler import CustomDDPMScheduler, AdaptiveNoiseScheduler
from src.utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        use_optimized_schedule: bool = True,
        schedule_config: Optional[NoiseScheduleConfig] = None,
        adaptation_frequency: int = 100,
    ):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(sample_size=32)

        # Use custom scheduler with optimized noise schedule
        if use_optimized_schedule:
            if schedule_config is None:
                schedule_config = NoiseScheduleConfig()

            # Initialize with adaptive scheduler
            self.scheduler = AdaptiveNoiseScheduler(
                num_train_timesteps=schedule_config.num_timesteps,
                beta_start=schedule_config.beta_start,
                beta_end=schedule_config.beta_end,
                beta_schedule=schedule_config.schedule_type,
                adaptation_frequency=adaptation_frequency,
            )

            # Store configuration for later optimization
            self.schedule_config = schedule_config
            self.use_optimized_schedule = True
        else:
            # Use standard scheduler
            self.scheduler = diffusers.schedulers.DDPMScheduler()
            self.use_optimized_schedule = False

        # Store training data for schedule optimization
        self.nex1_buffer = []
        self.nex6_buffer = []
        self.buffer_size = 1000  # Number of samples to keep for optimization

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (images.size(0),),
            device=self.device,
        )

        # Add noise using current schedule
        if self.use_optimized_schedule:
            noisy_images = self.scheduler.add_noise_with_schedule(images, noise, steps)
        else:
            noisy_images = self.scheduler.add_noise(images, noise, steps)

        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)

        # Log additional metrics if using optimized schedule
        if self.use_optimized_schedule:
            schedule_info = self.scheduler.get_noise_schedule_info()
            self.log("beta_mean", schedule_info["mean_beta"], prog_bar=True)
            self.log("beta_std", schedule_info["std_beta"], prog_bar=True)

            # Store samples for schedule optimization
            self._store_training_samples(images)

            # Adapt schedule periodically
            if batch_idx % self.scheduler.adaptation_frequency == 0:
                self._adapt_noise_schedule()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _store_training_samples(self, images: torch.Tensor):
        """Store training samples for schedule optimization."""
        # In practice, you would have separate NEX=1 and NEX=6 data
        # For now, we'll simulate this by treating current images as NEX=1
        # and creating NEX=6 by averaging multiple samples

        # Add to buffer
        self.nex1_buffer.append(images.detach().cpu())

        # Keep buffer size manageable
        if len(self.nex1_buffer) > self.buffer_size:
            self.nex1_buffer.pop(0)

    def _adapt_noise_schedule(self):
        """Adapt the noise schedule based on current training data."""
        if len(self.nex1_buffer) < 10:  # Need minimum samples
            return

        # Combine stored samples
        nex1_images = torch.cat(self.nex1_buffer[-10:], dim=0).to(self.device)

        # Create NEX=6 images by averaging (simulation)
        optimizer = NoiseScheduleOptimizer(self.schedule_config)
        nex6_images = optimizer.create_nex6_from_nex1(nex1_images)

        # Adapt schedule
        optimizer_config = {
            "num_timesteps": self.schedule_config.num_timesteps,
            "beta_start": self.schedule_config.beta_start,
            "beta_end": self.schedule_config.beta_end,
            "schedule_type": self.schedule_config.schedule_type,
            "target_snr_ratio": self.schedule_config.target_snr_ratio,
            "optimization_steps": 50,  # Shorter optimization during training
            "learning_rate": self.schedule_config.learning_rate,
        }

        self.scheduler.adapt_schedule(nex6_images, nex1_images, optimizer_config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def optimize_noise_schedule_offline(
        self, nex1_data: torch.Tensor, nex6_data: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Optimize noise schedule offline using provided data.

        Args:
            nex1_data: NEX=1 image data
            nex6_data: NEX=6 image data (if None, will be created from NEX=1)

        Returns:
            Dictionary containing optimization results
        """
        if not self.use_optimized_schedule:
            raise ValueError(
                "Model must be initialized with optimized schedule enabled"
            )

        # Create NEX=6 data if not provided
        if nex6_data is None:
            optimizer = NoiseScheduleOptimizer(self.schedule_config)
            nex6_data = optimizer.create_nex6_from_nex1(nex1_data)

        # Optimize schedule
        optimizer = NoiseScheduleOptimizer(self.schedule_config)
        optimized_schedule, history = optimizer.optimize_noise_schedule(
            nex6_data, nex1_data
        )

        # Update scheduler
        self.scheduler.set_custom_betas(optimized_schedule)

        return {
            "optimized_schedule": optimized_schedule,
            "optimization_history": history,
            "schedule_info": self.scheduler.get_noise_schedule_info(),
        }


class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(
            lambda sample: {"images": [self.augment(image) for image in sample["img"]]}
        )
        return torch.utils.data.DataLoader(
            dataset["train"], batch_size=128, shuffle=True, num_workers=4
        )


if __name__ == "__main__":
    model = DiffusionModel()
    data = DiffusionData()
    trainer = L.Trainer(max_epochs=150, precision="bf16-mixed")
    trainer.fit(model, data)
