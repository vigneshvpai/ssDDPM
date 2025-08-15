import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Optional, Tuple, Union
import numpy as np


class CustomDDPMScheduler(DDPMScheduler):
    """
    Custom DDPM Scheduler that uses optimized noise schedule based on SNR similarity.
    
    This scheduler extends the standard DDPMScheduler to allow for custom noise schedules
    that are optimized for specific data characteristics, particularly for medical imaging
    where SNR considerations are crucial.
    """
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 custom_betas: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Initialize custom DDPM scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Type of beta schedule ("linear", "cosine", "sigmoid")
            custom_betas: Custom beta values (if provided, overrides other parameters)
        """
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            **kwargs
        )
        
        # Override with custom betas if provided
        if custom_betas is not None:
            self.betas = custom_betas.to(self.device)
            self.num_train_timesteps = len(custom_betas)
            
            # Recalculate derived values
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
            
            # Calculate variance schedule
            self.variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
            self.log_variance = torch.log(torch.cat([torch.tensor([0.0]), self.variance[1:]]))
    
    def set_custom_betas(self, custom_betas: torch.Tensor):
        """
        Set custom beta values for the noise schedule.
        
        Args:
            custom_betas: Custom beta values tensor
        """
        self.betas = custom_betas.to(self.device)
        self.num_train_timesteps = len(custom_betas)
        
        # Recalculate derived values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculate variance schedule
        self.variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.log_variance = torch.log(torch.cat([torch.tensor([0.0]), self.variance[1:]]))
    
    def add_noise_with_schedule(self, 
                              original_samples: torch.FloatTensor,
                              noise: torch.FloatTensor,
                              timesteps: torch.IntTensor,
                              schedule_type: str = "linear") -> torch.FloatTensor:
        """
        Add noise to samples using specified schedule type.
        
        Args:
            original_samples: Original clean samples
            noise: Noise to add
            timesteps: Timesteps for each sample
            schedule_type: Type of noise schedule to use
            
        Returns:
            Noisy samples
        """
        # Get alpha values for current timesteps
        alpha_cumprod = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # Add noise according to DDPM formulation
        noisy_samples = torch.sqrt(alpha_cumprod) * original_samples + torch.sqrt(1 - alpha_cumprod) * noise
        
        return noisy_samples
    
    def get_noise_schedule_info(self) -> dict:
        """
        Get information about the current noise schedule.
        
        Returns:
            Dictionary containing schedule information
        """
        return {
            "num_timesteps": self.num_train_timesteps,
            "beta_start": self.betas[0].item(),
            "beta_end": self.betas[-1].item(),
            "mean_beta": self.betas.mean().item(),
            "std_beta": self.betas.std().item(),
            "final_alpha_cumprod": self.alphas_cumprod[-1].item(),
            "initial_alpha_cumprod": self.alphas_cumprod[0].item()
        }
    
    def evaluate_schedule_quality(self, 
                                clean_images: torch.Tensor,
                                noisy_images: torch.Tensor,
                                timesteps: torch.Tensor) -> dict:
        """
        Evaluate the quality of the noise schedule.
        
        Args:
            clean_images: Original clean images
            noisy_images: Noisy images
            timesteps: Timesteps used for noise addition
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate SNR at different timesteps
        snr_values = []
        for t in torch.unique(timesteps):
            mask = timesteps == t
            if mask.sum() > 0:
                clean_t = clean_images[mask]
                noisy_t = noisy_images[mask]
                noise_t = noisy_t - clean_t
                
                signal_power = torch.mean(clean_t ** 2)
                noise_power = torch.mean(noise_t ** 2)
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                snr_values.append(snr.item())
        
        return {
            "mean_snr": np.mean(snr_values) if snr_values else 0.0,
            "std_snr": np.std(snr_values) if snr_values else 0.0,
            "min_snr": np.min(snr_values) if snr_values else 0.0,
            "max_snr": np.max(snr_values) if snr_values else 0.0,
            "snr_values": snr_values
        }


class AdaptiveNoiseScheduler(CustomDDPMScheduler):
    """
    Adaptive noise scheduler that can adjust its schedule based on data characteristics.
    """
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 adaptation_frequency: int = 100,
                 **kwargs):
        """
        Initialize adaptive noise scheduler.
        
        Args:
            adaptation_frequency: How often to adapt the schedule (in training steps)
        """
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            **kwargs
        )
        
        self.adaptation_frequency = adaptation_frequency
        self.training_step = 0
        self.schedule_history = []
    
    def adapt_schedule(self, 
                      nex6_images: torch.Tensor,
                      nex1_images: torch.Tensor,
                      optimizer_config: dict):
        """
        Adapt the noise schedule based on current data.
        
        Args:
            nex6_images: High SNR images (NEX=6)
            nex1_images: Low SNR images (NEX=1)
            optimizer_config: Configuration for schedule optimization
        """
        from src.utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig
        
        # Create optimizer with current config
        config = NoiseScheduleConfig(**optimizer_config)
        optimizer = NoiseScheduleOptimizer(config)
        
        # Optimize schedule
        optimized_schedule, history = optimizer.optimize_noise_schedule(
            nex6_images, nex1_images, self.betas
        )
        
        # Update schedule
        self.set_custom_betas(optimized_schedule)
        
        # Record history
        self.schedule_history.append({
            "step": self.training_step,
            "schedule": optimized_schedule.clone(),
            "history": history
        })
        
        print(f"Adapted noise schedule at step {self.training_step}")
    
    def step(self, *args, **kwargs):
        """
        Override step method to track training progress.
        """
        self.training_step += 1
        return super().step(*args, **kwargs)
