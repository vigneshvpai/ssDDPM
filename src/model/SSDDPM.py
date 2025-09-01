import time
from diffusers import UNet2DModel, DDPMScheduler
import lightning as L
import torch
import torch.nn as nn
from src.model.ADC import ADC
from src.config.config import Config
from tqdm import tqdm


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
        self.adc_model = ADC()

        self.scheduler = DDPMScheduler(**Config.SSDDPM_CONFIG["SCHEDULER_CONFIG"])
        self.lambda_reg = Config.SSDDPM_CONFIG["lambda_reg"]
        self.num_inference_steps = Config.SSDDPM_CONFIG["num_inference_steps"]
        self.max_epochs = Config.SSDDPM_CONFIG["max_epochs"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), **Config.SSDDPM_CONFIG["OPTIMIZER_CONFIG"]
        )

        # Use cosine annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, batch):
        images, b_values, _ = batch  # Step 1: Sample batch y₀ ~ Y
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

        betas = self.scheduler.betas.to(steps.device)[steps].view(-1, 1, 1, 1)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(steps.device)[steps].view(
            -1, 1, 1, 1
        )

        y_prime_t_minus_1 = (1 / torch.sqrt(1 - betas)) * (
            noisy_images - (betas / torch.sqrt(1 - alphas_cumprod)) * residual
        ) + torch.sqrt(
            betas
        ) * epsilon_zero  # Step 7: y'_{t-1} = (1 / √1 - β_t) (y_t - (β_t / √1 - ā_t) ê_t) + √β_t ε₀

        S0_hat, D_hat = self.adc_model(
            y_prime_t_minus_1, b_values
        )  # Step 8: Ŝ₀, D̂ ← f_ADC(y'_{t-1})

        # Repeat each slice 25 times to match b_values shape
        S0_hat_expanded = S0_hat.repeat(1, 25, 1, 1)  # Shape: (2, 625, H, W)
        D_hat_expanded = D_hat.repeat(1, 25, 1, 1)  # Shape: (2, 625, H, W)
        # Reshape b_values to broadcast properly
        b_values_reshaped = b_values.view(2, 625, 1, 1)  # Shape: (2, 625, 1, 1)

        y_hat_t_minus_1 = S0_hat_expanded * torch.exp(
            -b_values_reshaped * D_hat_expanded
        )  # Step 9: ŷ_{t-1} ← Ŝ₀ e^(-b D̂)

        noise_loss = torch.nn.functional.mse_loss(residual, noise)  # ||ê_t - ε||²₂
        reg_loss = torch.nn.functional.mse_loss(
            y_hat_t_minus_1, y_prime_t_minus_1
        )  # Self-supervised: ||ŷ_{t-1} - f₀(ŷ_{t-1}, t)||²₂

        loss = (
            noise_loss + self.lambda_reg * reg_loss
        )  # Total loss: noise loss + self-supervised reg loss

        return loss

    @torch.no_grad()
    def inference(self, y_hat_t, b_values):
        self.eval()

        print(f"Starting inference with {self.num_inference_steps} steps...")

        start_time = time.time()

        # Create progress bar
        pbar = tqdm(
            range(self.num_inference_steps - 1, -1, -1),
            desc="Inference Progress",
            total=self.num_inference_steps,
            unit="step",
        )

        for t in pbar:
            pbar.set_description(
                f"Step {self.num_inference_steps - 1 - t + 1}/{self.num_inference_steps} (t={t})"
            )

            # Step 2: ê_t ← f_0(ŷ_t, t)
            timesteps = torch.full(
                (y_hat_t.shape[0],), t, device=y_hat_t.device, dtype=torch.long
            )
            residual = self.model(y_hat_t, timesteps).sample

            # Step 3: ε_0 ~ N(0, I)
            epsilon_zero = torch.randn_like(y_hat_t)

            # Step 4: y'_t-1 ← (1 / √(1 - β_t)) * (ŷ_t - (β_t / √(1 - α_t)) * ê_t) + √(β_t) * ε_0
            beta_t = self.scheduler.betas[t].to(y_hat_t.device).view(-1, 1, 1, 1)
            alpha_cumprod_t = (
                self.scheduler.alphas_cumprod[t].to(y_hat_t.device).view(-1, 1, 1, 1)
            )
            y_prime_t_minus_1 = (1 / torch.sqrt(1 - beta_t)) * (
                y_hat_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * residual
            ) + torch.sqrt(beta_t) * epsilon_zero

            # Step 5: Ŝ_0, D̂ ← f_ADC(y'_t-1)
            S0_hat, D_hat = self.adc_model(y_prime_t_minus_1, b_values)

            # Step 6: ŷ_t-1 ← Ŝ_0 * e^(-b * D̂)
            # Handle the complex reshaping for your specific data format
            S0_hat_expanded = S0_hat.repeat(1, 25, 1, 1)
            D_hat_expanded = D_hat.repeat(1, 25, 1, 1)
            b_values_reshaped = b_values.view(b_values.shape[0], 625, 1, 1)
            y_hat_t_minus_1 = S0_hat_expanded * torch.exp(
                -b_values_reshaped * D_hat_expanded
            )
            # Update for next iteration
            y_hat_t = y_hat_t_minus_1

        # Calculate total time
        total_time = time.time() - start_time

        print(
            f"Inference completed! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        # Return ŷ_0
        return y_hat_t

    def training_step(self, batch):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)
        return loss
