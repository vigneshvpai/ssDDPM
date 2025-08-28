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
        self.lambda_reg = Config.LAMBDA_REG

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **Config.OPTIMIZER_CONFIG)

        # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        images, b_values = batch  # Step 1: Sample batch y₀ ~ Y
        noise = torch.randn_like(images)  # Step 3: Sample ε ~ N(0, I)
        steps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )  # Step 2: Sample t ~ Uniform({1, ..., T})
        noisy_images = self.scheduler.add_noise(
            images, noise, steps
        )  # Step 4: y_t = √ā_t y₀ + √1 - ā_t ε
        residual = self.model(noisy_images, steps).sample  # Step 5: ê_t = f₀(y_t, t)
        epsilon_zero = torch.randn_like(images)  # Step 6: ε₀ ~ N(0, I)

        y_prime_t_minus_1 = (1 / torch.sqrt(1 - self.scheduler.betas[steps])) * (
            noisy_images  # y_t term
            - (
                self.scheduler.betas[steps]
                / torch.sqrt(1 - self.scheduler.alphas_cumprod[steps])
            )
            * residual  # ê_t term
        ) + torch.sqrt(
            self.scheduler.betas[steps]
        ) * epsilon_zero  # Step 7: y'_{t-1} = (1 / √1 - β_t) (y_t - (β_t / √1 - α_t) ê_t) + √β_t ε₀

        S0_hat, D_hat = self.f_ADC(y_prime_t_minus_1)  # Step 8: Ŝ₀, D̂ ← f_ADC(y'_{t-1})

        y_hat_t_minus_1 = S0_hat * torch.exp(
            -self.b * D_hat
        )  # Step 9: ŷ_{t-1} ← Ŝ₀ e^(-b D̂)

        noise_loss = torch.nn.functional.mse_loss(residual, noise)  # ||ê_t - ε||²₂
        reg_loss = torch.nn.functional.mse_loss(
            y_hat_t_minus_1, y_prime_t_minus_1
        )  # ||ŷ_{t-1} - y'_{t-1}||²₂
        loss = (
            noise_loss + self.lambda_reg * reg_loss
        )  # Step 10: ||ê_t - ε||²₂ + λ ||ŷ_{t-1} - y'_{t-1}||²₂

        return loss
