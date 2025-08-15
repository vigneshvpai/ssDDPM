import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class NoiseScheduleConfig:
    """Configuration for noise schedule optimization"""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "linear"  # "linear", "cosine", "sigmoid"
    target_snr_ratio: float = 1.0  # Target SNR ratio between NEX=6 and NEX=1
    optimization_steps: int = 100
    learning_rate: float = 1e-3


class NoiseScheduleOptimizer:
    """
    Optimizes noise schedule based on SNR similarity between NEX=6 and NEX=1 images.
    
    This implementation follows the approach described in the paper where the optimal
    noise schedule is chosen to maximize SNR similarity between noise-corrupted NEX=6
    images and NEX=1 samples at t=T.
    """
    
    def __init__(self, config: NoiseScheduleConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_noise_schedule(self, schedule_type: str = None) -> torch.Tensor:
        """
        Create noise schedule with specified type.
        
        Args:
            schedule_type: Type of schedule ("linear", "cosine", "sigmoid")
            
        Returns:
            Tensor of shape (num_timesteps,) containing beta values
        """
        schedule_type = schedule_type or self.config.schedule_type
        
        if schedule_type == "linear":
            return torch.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                self.config.num_timesteps
            )
        elif schedule_type == "cosine":
            # Cosine schedule as in Improved DDPM
            steps = self.config.num_timesteps + 1
            x = torch.linspace(0, self.config.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.config.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        elif schedule_type == "sigmoid":
            # Sigmoid schedule for more gradual noise addition
            x = torch.linspace(-6, 6, self.config.num_timesteps)
            betas = torch.sigmoid(x) * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
            return betas
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def calculate_snr(self, image: torch.Tensor, noise: torch.Tensor) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Args:
            image: Clean image tensor
            noise: Noise tensor
            
        Returns:
            SNR value
        """
        signal_power = torch.mean(image ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr.item()
    
    def add_noise_to_image(self, image: torch.Tensor, timestep: int, 
                          noise_schedule: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to image according to noise schedule.
        
        Args:
            image: Clean image tensor
            timestep: Current timestep
            noise_schedule: Beta values for noise schedule
            
        Returns:
            Tuple of (noisy_image, noise)
        """
        # Calculate cumulative alphas
        alphas = 1.0 - noise_schedule
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Get alpha for current timestep
        alpha_t = alphas_cumprod[timestep]
        
        # Generate noise
        noise = torch.randn_like(image)
        
        # Add noise according to DDPM formulation
        noisy_image = torch.sqrt(alpha_t) * image + torch.sqrt(1 - alpha_t) * noise
        
        return noisy_image, noise
    
    def evaluate_noise_schedule(self, nex6_images: torch.Tensor, nex1_images: torch.Tensor,
                              noise_schedule: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate noise schedule by comparing SNR between NEX=6 and NEX=1 at t=T.
        
        Args:
            nex6_images: High SNR images (NEX=6)
            nex1_images: Low SNR images (NEX=1)
            noise_schedule: Beta values for noise schedule
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Add noise to NEX=6 images at t=T (maximum noise)
        t_max = self.config.num_timesteps - 1
        noisy_nex6, noise_nex6 = self.add_noise_to_image(nex6_images, t_max, noise_schedule)
        
        # Calculate SNR for noisy NEX=6 and original NEX=1
        snr_noisy_nex6 = self.calculate_snr(nex6_images, noise_nex6)
        snr_nex1 = self.calculate_snr(nex1_images, torch.zeros_like(nex1_images))  # Clean NEX=1
        
        # Calculate SNR ratio
        snr_ratio = snr_noisy_nex6 / (snr_nex1 + 1e-8)
        
        # Calculate similarity metric (closer to 1 is better)
        similarity = 1.0 / (1.0 + abs(snr_ratio - self.config.target_snr_ratio))
        
        return {
            "snr_noisy_nex6": snr_noisy_nex6,
            "snr_nex1": snr_nex1,
            "snr_ratio": snr_ratio,
            "similarity": similarity,
            "target_similarity": 1.0
        }
    
    def optimize_noise_schedule(self, nex6_images: torch.Tensor, nex1_images: torch.Tensor,
                              initial_schedule: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Optimize noise schedule to maximize SNR similarity.
        
        Args:
            nex6_images: High SNR images (NEX=6)
            nex1_images: Low SNR images (NEX=1)
            initial_schedule: Initial noise schedule (optional)
            
        Returns:
            Tuple of (optimized_schedule, optimization_history)
        """
        if initial_schedule is None:
            initial_schedule = self.create_noise_schedule()
        
        # Convert to parameter for optimization
        schedule_param = torch.nn.Parameter(initial_schedule.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([schedule_param], lr=self.config.learning_rate)
        
        history = {
            "losses": [],
            "similarities": [],
            "snr_ratios": []
        }
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Ensure schedule values are valid (between 0 and 1)
            schedule = torch.sigmoid(schedule_param) * 0.02  # Scale to reasonable beta range
            
            # Evaluate current schedule
            metrics = self.evaluate_noise_schedule(nex6_images, nex1_images, schedule)
            
            # Loss is negative similarity (we want to maximize similarity)
            loss = -metrics["similarity"]
            
            # Add regularization to prevent extreme values
            reg_loss = 0.01 * torch.mean(schedule ** 2)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Record history
            history["losses"].append(total_loss.item())
            history["similarities"].append(metrics["similarity"])
            history["snr_ratios"].append(metrics["snr_ratio"])
            
            if step % 10 == 0:
                print(f"Step {step}: Loss={total_loss.item():.4f}, "
                      f"Similarity={metrics['similarity']:.4f}, "
                      f"SNR Ratio={metrics['snr_ratio']:.4f}")
        
        # Return final optimized schedule
        final_schedule = torch.sigmoid(schedule_param) * 0.02
        return final_schedule, history
    
    def visualize_noise_schedule(self, schedule: torch.Tensor, 
                               save_path: Optional[str] = None) -> None:
        """
        Visualize the noise schedule.
        
        Args:
            schedule: Beta values for noise schedule
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot beta values
        plt.subplot(2, 2, 1)
        plt.plot(schedule.detach().cpu().numpy())
        plt.title("Noise Schedule (Beta Values)")
        plt.xlabel("Timestep")
        plt.ylabel("Beta")
        plt.grid(True)
        
        # Plot cumulative alphas
        alphas = 1.0 - schedule
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        plt.subplot(2, 2, 2)
        plt.plot(alphas_cumprod.detach().cpu().numpy())
        plt.title("Cumulative Alpha Values")
        plt.xlabel("Timestep")
        plt.ylabel("Alpha_cumprod")
        plt.grid(True)
        
        # Plot noise level at each timestep
        noise_levels = torch.sqrt(1 - alphas_cumprod)
        plt.subplot(2, 2, 3)
        plt.plot(noise_levels.detach().cpu().numpy())
        plt.title("Noise Level at Each Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Noise Level")
        plt.grid(True)
        
        # Plot signal level at each timestep
        signal_levels = torch.sqrt(alphas_cumprod)
        plt.subplot(2, 2, 4)
        plt.plot(signal_levels.detach().cpu().numpy())
        plt.title("Signal Level at Each Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Signal Level")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_nex6_from_nex1(self, nex1_images: torch.Tensor, num_repetitions: int = 6) -> torch.Tensor:
        """
        Create NEX=6 images by averaging multiple NEX=1 repetitions.
        
        Args:
            nex1_images: NEX=1 images tensor
            num_repetitions: Number of repetitions to average
            
        Returns:
            Averaged NEX=6 images
        """
        # For simulation, we'll add different noise realizations and average them
        # In practice, this would be actual repeated acquisitions
        batch_size = nex1_images.shape[0]
        nex6_images = torch.zeros_like(nex1_images)
        
        for i in range(num_repetitions):
            # Add different noise realizations
            noise = torch.randn_like(nex1_images) * 0.1  # Simulate acquisition noise
            nex6_images += nex1_images + noise
        
        return nex6_images / num_repetitions


def optimize_noise_schedule_for_data(nex1_data: torch.Tensor, 
                                   config: NoiseScheduleConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convenience function to optimize noise schedule for given data.
    
    Args:
        nex1_data: NEX=1 image data
        config: Configuration for optimization
        
    Returns:
        Tuple of (optimized_schedule, optimization_history)
    """
    optimizer = NoiseScheduleOptimizer(config)
    
    # Create NEX=6 data from NEX=1
    nex6_data = optimizer.create_nex6_from_nex1(nex1_data)
    
    # Optimize noise schedule
    optimized_schedule, history = optimizer.optimize_noise_schedule(nex6_data, nex1_data)
    
    return optimized_schedule, history
