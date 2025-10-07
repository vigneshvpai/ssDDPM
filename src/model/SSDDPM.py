import time
from diffusers import UNet2DModel, DDPMScheduler
import lightning as L
import torch
import numpy as np
from src.model.ADC import ADC
from src.config.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from src.data.Preprocess import Preprocess
from src.data.Postprocess import Postprocess


class SSDDPM(L.LightningModule):
    def __init__(self, in_channels, out_channels, run_name):
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
            block_out_channels=(
                64,
                128,
                128,
                256,
            ),
            layers_per_block=2,
        )
        self.adc_model = ADC()

        self.scheduler = DDPMScheduler(**Config.SSDDPM_CONFIG["SCHEDULER_CONFIG"])
        self.lambda_adc = Config.SSDDPM_CONFIG["lambda_adc"]
        self.num_inference_steps = Config.SSDDPM_CONFIG["num_inference_steps"]
        self.max_epochs = Config.SSDDPM_CONFIG["max_epochs"]
        self.n_slices = Config.DWI_CONFIG["n_slices"]
        self.n_bvals = Config.DWI_CONFIG["n_bvals"]
        self.run_name = run_name
        self.enable_progress_bar = Config.LOGGER_CONFIG["enable_progress_bar"]

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

    def _get_y_prime_t_minus_1(
        self, noisy_images, residual, betas, alphas_cumprod, mode="train"
    ):
        if mode == "train":
            epsilon_zero = torch.randn_like(noisy_images)  # Step 6: ε₀ ~ N(0, I)

            y_prime_t_minus_1 = (1 / torch.sqrt(1 - betas)) * (
                noisy_images - (betas / torch.sqrt(1 - alphas_cumprod)) * residual
            ) + torch.sqrt(
                betas
            ) * epsilon_zero  # Step 7: y'_{t-1} = (1 / √1 - β_t) (y_t - (β_t / √1 - ā_t) ê_t) + √β_t ε₀
        else:
            y_prime_t_minus_1 = (1 / torch.sqrt(1 - betas)) * (
                noisy_images - (betas / torch.sqrt(1 - alphas_cumprod)) * residual
            )

        return y_prime_t_minus_1

    def _get_y_hat_t_minus_1(self, S0_hat, D_hat, b_values):
        # Get dimensions dynamically
        batch_size, width, height = S0_hat.shape  # S0_hat: (B, H, W)
        n_bvals = b_values.shape[1]  # Get n_bvals from b_values

        # Expand S0_hat and D_hat to include n_bvals dimension
        # S0_hat: (B, W, H) -> (B, 1, W, H)
        S0_hat_expanded = S0_hat.unsqueeze(1)
        # D_hat: (B, W, H) -> (B, 1, W, H)
        D_hat_expanded = D_hat.unsqueeze(1)

        # Reshape b_values to broadcast properly
        # b_values: (batch_size, n_bvals) -> (batch_size, n_bvals, 1, 1)
        b_values_reshaped = b_values.view(batch_size, n_bvals, 1, 1)

        # Apply the mono-exponential model: S(b) = S0 * exp(-b * ADC)
        y_hat_t_minus_1 = S0_hat_expanded * torch.exp(
            -b_values_reshaped * D_hat_expanded
        )

        return y_hat_t_minus_1

    def _log_specific_slice(
        self,
        images,
        b_values,
        other_info,
        step_or_epoch,
        prefix="train",
        save_dir="train_images",
    ):
        images_cpu = images.cpu().detach()
        b_values_cpu = b_values.cpu().detach()

        # Detect if images are normalized or in true signal range
        image_min, image_max = images_cpu.min(), images_cpu.max()
        is_normalized = image_min >= -1.1 and image_max <= 1.1  # Allow small tolerance

        slice_values = other_info["slice"]
        middle_slice_mask = slice_values == (self.n_slices // 2)
        middle_slice_indices = torch.where(middle_slice_mask)[0].tolist()

        if not middle_slice_indices:
            return

        # Create directories
        epoch_dir = os.path.join(save_dir, f"epoch_{step_or_epoch:03d}")
        plots_dir = os.path.join(epoch_dir, "plots")
        pt_files_dir = os.path.join(epoch_dir, "pt_files")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(pt_files_dir, exist_ok=True)

        for _, image_idx in enumerate(middle_slice_indices):
            single_image = images_cpu[image_idx : image_idx + 1]
            single_b_values = b_values_cpu[image_idx : image_idx + 1]
            single_info = {key: other_info[key][image_idx] for key in other_info}

            # Save metadata including range info
            pt_filename = os.path.join(
                pt_files_dir, f"{single_info['original_filename']}.pt"
            )
            torch.save(
                {
                    "image": single_image,
                    "b_values": single_b_values,
                    "other_info": single_info,
                    "step_or_epoch": step_or_epoch,
                    "prefix": prefix,
                    "is_normalized": is_normalized,  # Save range info
                    "image_range": (float(image_min), float(image_max)),
                },
                pt_filename,
            )

            # Create plot with adaptive scaling
            if self.n_bvals == 3:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            else:
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            axes_flat = axes.flatten()

            for i in range(self.n_bvals):
                b_value_image = single_image[0, i, :, :].float().numpy().T[::-1, :]

                # Adaptive display scaling
                if is_normalized:
                    # For normalized images, use fixed [-1, 1] range
                    vmin, vmax = -1, 1
                    title_suffix = " (norm)"
                else:
                    # For true signals, use percentiles to avoid outlier effects
                    vmin = np.percentile(b_value_image, 1)
                    vmax = np.percentile(b_value_image, 99)
                    title_suffix = " (raw)"

                im = axes_flat[i].imshow(
                    b_value_image, cmap="gray", vmin=vmin, vmax=vmax
                )
                axes_flat[i].set_title(
                    f"B-value: {int(single_b_values[0, i])}{title_suffix}", fontsize=8
                )
                axes_flat[i].axis("off")

                # Optional: Add colorbar for reference
                # plt.colorbar(im, ax=axes_flat[i], fraction=0.046)

            plt.savefig(
                os.path.join(plots_dir, f"{single_info['original_filename']}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Explicit cleanup
            del single_image, single_b_values, b_value_image
            del fig, axes, axes_flat

    def compute_loss(self, batch, mode="train"):
        original_images, b_values, other_info = batch  # Step 1: Sample batch y₀ ~ Y

        images_normalized, min_val, max_val = Preprocess.normalize_to_minus_1_1(
            original_images
        )
        images_normalized_padded = Preprocess.pad_to_unet_compatible(images_normalized)

        noise, steps = self._get_noise_and_timesteps(images_normalized_padded)

        y_t_normalized_padded = self.scheduler.add_noise(
            images_normalized_padded, noise, steps
        )  # Step 4: y_t = √ā_t y₀ + √1 - ā_t ε

        residual = self.model(
            y_t_normalized_padded, steps
        ).sample  # Step 5: ê_t = f₀(y_t, t)
        noise_loss = torch.nn.functional.mse_loss(
            Postprocess.unpad_from_unet_compatible(residual),
            Postprocess.unpad_from_unet_compatible(noise),
        )  # ||ê_t - ε||²₂
        self.log(
            f"{mode}_noise_loss",
            noise_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=Config.BATCH_SIZE,
        )

        # betas, alphas_cumprod = self._get_beta_and_alpha_cumprod(steps)

        # y_prime_t_minus_1_normalized_padded = self._get_y_prime_t_minus_1(
        #     y_t_normalized_padded, residual, betas, alphas_cumprod
        # )
        # y_prime_t_minus_1_normalized = Postprocess.unpad_from_unet_compatible(
        #     y_prime_t_minus_1_normalized_padded
        # )
        # y_prime_t_minus_1 = Postprocess.denormalize_from_minus_1_1(
        #     y_prime_t_minus_1_normalized, min_val, max_val
        # )

        # S0_hat, D_hat = self.adc_model(
        #     y_prime_t_minus_1, b_values
        # )  # Step 8: Ŝ₀, D̂ ← f_ADC(y'_{t-1})

        # y_hat_t_minus_1 = self._get_y_hat_t_minus_1(S0_hat, D_hat, b_values)

        # # Normalize both to [-1,1] for comparable loss scaling
        # y_prime_norm, _, _ = Preprocess.normalize_to_minus_1_1(y_prime_t_minus_1)
        # y_hat_norm, _, _ = Preprocess.normalize_to_minus_1_1(y_hat_t_minus_1)
        # adc_loss = torch.nn.functional.mse_loss(
        #     y_hat_norm,
        #     y_prime_norm,
        # )  # Self-supervised: ||ŷ_{t-1} - f₀(ŷ_{t-1}, t)||²₂
        # self.log(
        #     f"{mode}_adc_loss",
        #     adc_loss,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=Config.BATCH_SIZE,
        # )

        # total_loss = (
        #     noise_loss + self.lambda_adc * adc_loss
        # )  # Total loss: noise loss + ADC loss

        total_loss = noise_loss

        self.log(
            f"{mode}_total_loss",
            total_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=Config.BATCH_SIZE,
        )

        # loss_ratio = (self.lambda_adc * adc_loss) / (noise_loss + 1e-8)
        # self.log(
        #     f"{mode}_loss_ratio",
        #     loss_ratio,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=Config.BATCH_SIZE,
        # )

        if mode == "val" and (
            self.current_epoch % Config.CHECKPOINT_CONFIG["every_n_epochs"] == 0
        ):
            if self.current_epoch == 0:
                self._log_specific_slice(
                    original_images,
                    b_values,
                    other_info,
                    step_or_epoch=self.current_epoch,
                    prefix=mode,
                    save_dir=f"{mode}_images/{self.run_name}/original_images",
                )
            self._log_specific_slice(
                residual,
                b_values,
                other_info,
                step_or_epoch=self.current_epoch,
                prefix=mode,
                save_dir=f"{mode}_images/{self.run_name}/residual_images",
            )
            # self._log_specific_slice(
            #     y_prime_t_minus_1,
            #     b_values,
            #     other_info,
            #     step_or_epoch=self.current_epoch,
            #     prefix=mode,
            #     save_dir=f"{mode}_images/{self.run_name}/y_prime_t_minus_1",
            # )
            # self._log_specific_slice(
            #     y_hat_t_minus_1,
            #     b_values,
            #     other_info,
            #     step_or_epoch=self.current_epoch,
            #     prefix=mode,
            #     save_dir=f"{mode}_images/{self.run_name}/y_hat_t_minus_1",
            # )
            # if self.current_epoch != 0:
            #     denoised_images = self.inference(noisy_images, b_values)
            #     self._log_specific_slice(
            #         denoised_images,
            #         b_values,
            #         other_info,
            #         step_or_epoch=self.current_epoch,
            #         prefix=mode,
            #         save_dir=f"{mode}_images/{self.run_name}/denoised_images",
            #     )

        return total_loss

    @torch.no_grad()
    def inference(self, y_hat_t, b_values):
        self.eval()

        # Set the scheduler timesteps for inference
        self.scheduler.set_timesteps(self.num_inference_steps)

        print(f"Starting inference with {self.num_inference_steps} steps...")

        start_time = time.time()

        # Create progress bar
        if self.enable_progress_bar:
            pbar = tqdm(
                self.scheduler.timesteps,
                desc="Inference Progress",
                total=self.num_inference_steps,
                unit="step",
            )
        else:
            pbar = self.scheduler.timesteps

        for i, t in enumerate(pbar):
            if self.enable_progress_bar:
                pbar.set_description(f"Step {i + 1}/{self.num_inference_steps} (t={t})")

            # Create timestep tensor for the model
            timesteps = torch.full(
                (y_hat_t.shape[0],), t, device=y_hat_t.device, dtype=torch.long
            )

            residual = self.model(y_hat_t, timesteps).sample

            beta_t, alpha_cumprod_t = self._get_beta_and_alpha_cumprod(timesteps)

            # Step 4: y'_t-1 ← (1 / √(1 - β_t)) * (ŷ_t - (β_t / √(1 - α_t)) * ê_t) + √(β_t) * ε_0
            y_prime_t_minus_1 = self._get_y_prime_t_minus_1(
                y_hat_t, residual, beta_t, alpha_cumprod_t, mode="inference"
            )

            # Step 5: Ŝ_0, D̂ ← f_ADC(y'_t-1)
            S0_hat, D_hat = self.adc_model(y_prime_t_minus_1, b_values)

            # Step 6: ŷ_t-1 ← Ŝ_0 * e^(-b * D̂)
            y_hat_t_minus_1 = self._get_y_hat_t_minus_1(S0_hat, D_hat, b_values)

            # Update for next iteration
            y_hat_t = y_hat_t_minus_1

            del (
                timesteps,
                beta_t,
                alpha_cumprod_t,
                residual,
                y_prime_t_minus_1,
                S0_hat,
                D_hat,
            )

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
        loss = self.compute_loss(batch, mode="val")
        return loss
