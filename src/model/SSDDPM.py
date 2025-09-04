import time
from diffusers import UNet2DModel, DDPMScheduler
import lightning as L
import torch
import torch.nn as nn
from src.model.ADC import ADC
from src.config.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


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
        self.lambda_adc = Config.SSDDPM_CONFIG["lambda_adc"]
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

    def _get_noise_and_timesteps(self, images):
        noise = torch.randn_like(images)  # Step 3: Sample ε ~ N(0, I)
        steps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )  # Step 2: Sample t ~ Uniform({1, ..., T})

        return noise, steps

    def _get_beta_and_alpha_cumprod(self, steps):
        betas = self.scheduler.betas.to(steps.device)[steps].view(-1, 1, 1, 1)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(steps.device)[steps].view(
            -1, 1, 1, 1
        )

        return betas, alphas_cumprod

    def _get_y_prime_t_minus_1(self, noisy_images, residual, betas, alphas_cumprod):
        epsilon_zero = torch.randn_like(noisy_images)  # Step 6: ε₀ ~ N(0, I)

        y_prime_t_minus_1 = (1 / torch.sqrt(1 - betas)) * (
            noisy_images - (betas / torch.sqrt(1 - alphas_cumprod)) * residual
        ) + torch.sqrt(
            betas
        ) * epsilon_zero  # Step 7: y'_{t-1} = (1 / √1 - β_t) (y_t - (β_t / √1 - ā_t) ê_t) + √β_t ε₀

        return y_prime_t_minus_1

    def _get_y_hat_t_minus_1(self, S0_hat, D_hat, b_values):
        # Repeat each slice 25 times to match b_values shape
        S0_hat_expanded = S0_hat.repeat_interleave(25, dim=1)  # Shape: (2, 625, H, W)
        D_hat_expanded = D_hat.repeat_interleave(25, dim=1)  # Shape: (2, 625, H, W)
        # Reshape b_values to broadcast properly
        b_values_reshaped = b_values.view(2, 625, 1, 1)  # Shape: (2, 625, 1, 1)

        y_hat_t_minus_1 = S0_hat_expanded * torch.exp(
            -b_values_reshaped * D_hat_expanded
        )  # Step 9: ŷ_{t-1} ← Ŝ₀ e^(-b D̂)

        return y_hat_t_minus_1

    def _log_specific_slice(
        self, images, step_or_epoch, prefix="train", save_dir="training_images"
    ):

        # Just take slice 11 directly
        slice_12 = images[0, 11, :, :]  # Slice 12

        self.logger.experiment.add_image(
            f"{prefix}/slice", slice_12, step_or_epoch, dataformats="HW"
        )

        # Also save to disk using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_12.cpu().detach().numpy(), cmap="gray")
        plt.title(f"{prefix} - Epoch {step_or_epoch}")
        plt.axis("off")

        # Create directory if it doesn't exist
        save_dir = f"{save_dir}/{prefix.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot
        plt.savefig(
            f"{save_dir}/epoch_{step_or_epoch:03d}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()  # Close to free memory

    def compute_loss(self, batch):
        images, b_values, _ = batch  # Step 1: Sample batch y₀ ~ Y

        noise, steps = self._get_noise_and_timesteps(images)

        noisy_images = self.scheduler.add_noise(
            images, noise, steps
        )  # Step 4: y_t = √ā_t y₀ + √1 - ā_t ε

        residual = self.model(noisy_images, steps).sample  # Step 5: ê_t = f₀(y_t, t)

        betas, alphas_cumprod = self._get_beta_and_alpha_cumprod(steps)

        y_prime_t_minus_1 = self._get_y_prime_t_minus_1(
            noisy_images, residual, betas, alphas_cumprod
        )

        S0_hat, D_hat = self.adc_model(
            y_prime_t_minus_1, b_values
        )  # Step 8: Ŝ₀, D̂ ← f_ADC(y'_{t-1})

        y_hat_t_minus_1 = self._get_y_hat_t_minus_1(S0_hat, D_hat, b_values)

        noise_loss = torch.nn.functional.mse_loss(residual, noise)  # ||ê_t - ε||²₂
        self.log("train/noise_loss", noise_loss, on_epoch=True)
        adc_loss = torch.nn.functional.mse_loss(
            y_hat_t_minus_1, y_prime_t_minus_1
        )  # Self-supervised: ||ŷ_{t-1} - f₀(ŷ_{t-1}, t)||²₂
        self.log("train/adc_loss", adc_loss, on_epoch=True)
        loss = (
            noise_loss + self.lambda_adc * adc_loss
        )  # Total loss: noise loss + self-supervised reg loss
        self.log("train/total_loss", loss, on_epoch=True)

        if self.global_step % self.trainer.num_training_batches == 0:
            self._log_specific_slice(images, self.current_epoch, "train/orignal")
            self._log_specific_slice(
                noisy_images, self.current_epoch, "train/noisy_image"
            )
            self._log_specific_slice(residual, self.current_epoch, "train/residual")
            self._log_specific_slice(
                y_prime_t_minus_1, self.current_epoch, "train/y_prime_t_minus_1"
            )
            self._log_specific_slice(
                y_hat_t_minus_1, self.current_epoch, "train/y_hat_t_minus_1"
            )

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

            _, timesteps = self._get_noise_and_timesteps(
                y_hat_t
            )  # Step 2: ê_t ← f_0(ŷ_t, t)

            residual = self.model(y_hat_t, timesteps).sample

            beta_t, alpha_cumprod_t = self._get_beta_and_alpha_cumprod(timesteps)

            # Step 4: y'_t-1 ← (1 / √(1 - β_t)) * (ŷ_t - (β_t / √(1 - α_t)) * ê_t) + √(β_t) * ε_0
            y_prime_t_minus_1 = self._get_y_prime_t_minus_1(
                y_hat_t, residual, beta_t, alpha_cumprod_t
            )

            # Step 5: Ŝ_0, D̂ ← f_ADC(y'_t-1)
            S0_hat, D_hat = self.adc_model(y_prime_t_minus_1, b_values)

            # Step 6: ŷ_t-1 ← Ŝ_0 * e^(-b * D̂)
            y_hat_t_minus_1 = self._get_y_hat_t_minus_1(S0_hat, D_hat, b_values)

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
        return loss

    def validation_step(self, batch):
        images, b_values, _ = batch

        # Original noisy image generation (same as training)
        noise, steps = self._get_noise_and_timesteps(images)
        noisy_images = self.scheduler.add_noise(images, noise, steps)

        # Self-supervised validation: add additional synthetic noise
        # Sample additional noise for validation
        additional_noise = torch.randn_like(noisy_images)
        additional_steps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )

        # Create "noisier" images by adding synthetic noise
        noisier_images = self.scheduler.add_noise(
            noisy_images, additional_noise, additional_steps
        )

        # Use noisier_images as input, treat original noisy_images as target
        residual = self.model(noisier_images, additional_steps).sample

        # Get beta and alpha for the additional noise steps
        betas_syn, alphas_cumprod_syn = self._get_beta_and_alpha_cumprod(
            additional_steps
        )

        # Compute the predicted denoised version
        y_prime_t_minus_1_syn = self._get_y_prime_t_minus_1(
            noisier_images, residual, betas_syn, alphas_cumprod_syn
        )

        # ADC model prediction on the synthetic noise denoised version
        S0_hat_syn, D_hat_syn = self.adc_model(y_prime_t_minus_1_syn, b_values)
        y_hat_t_minus_1_syn = self._get_y_hat_t_minus_1(S0_hat_syn, D_hat_syn, b_values)

        # Self-supervised loss: compare predicted denoised image with original noisy image
        # This tests if the model can remove the added synthetic noise and return to the original noisy state
        noise_loss_syn = torch.nn.functional.mse_loss(residual, additional_noise)
        self.log("val/noise_loss", noise_loss_syn, on_epoch=True)
        adc_loss_syn = torch.nn.functional.mse_loss(
            y_hat_t_minus_1_syn, y_prime_t_minus_1_syn
        )
        self.log("val/adc_loss", adc_loss_syn, on_epoch=True)

        # Total self-supervised validation loss
        val_loss = noise_loss_syn + self.lambda_adc * adc_loss_syn

        # Log the synthetic noise validation loss
        self.log("val/total_loss", val_loss, on_epoch=True)

        if self.global_step % self.trainer.num_training_batches == 0:
            self._log_specific_slice(
                images, self.current_epoch, "val/original", save_dir="validation_images"
            )
            self._log_specific_slice(
                noisy_images,
                self.current_epoch,
                "val/noisy_image",
                save_dir="validation_images",
            )
            self._log_specific_slice(
                noisier_images,
                self.current_epoch,
                "val/noisier_image",
                save_dir="validation_images",
            )
            self._log_specific_slice(
                residual,
                self.current_epoch,
                "val/residual",
                save_dir="validation_images",
            )
            self._log_specific_slice(
                y_prime_t_minus_1_syn,
                self.current_epoch,
                "val/y_prime_t_minus_1",
                save_dir="validation_images",
            )
            self._log_specific_slice(
                y_hat_t_minus_1_syn,
                self.current_epoch,
                "val/y_hat_t_minus_1",
                save_dir="validation_images",
            )
        return val_loss
