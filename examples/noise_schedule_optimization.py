#!/usr/bin/env python3
"""
Example script demonstrating noise schedule optimization for medical imaging.

This script shows how to:
1. Optimize noise schedule based on SNR similarity between NEX=6 and NEX=1 images
2. Train a diffusion model with the optimized schedule
3. Evaluate the performance of different noise schedules
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig
from models.ssddpm import DiffusionModel
from models.custom_scheduler import CustomDDPMScheduler


def create_synthetic_medical_data(num_samples: int = 100, image_size: int = 64) -> tuple:
    """
    Create synthetic medical imaging data for demonstration.
    
    In practice, this would be replaced with actual DWI lesion assessment data.
    
    Args:
        num_samples: Number of samples to create
        image_size: Size of the images
        
    Returns:
        Tuple of (nex1_images, nex6_images)
    """
    # Create synthetic brain-like structures
    nex1_images = torch.zeros(num_samples, 1, image_size, image_size)
    
    for i in range(num_samples):
        # Create random brain-like structures
        # Simulate different tissue types with varying intensities
        background = torch.randn(image_size, image_size) * 0.1
        white_matter = torch.randn(image_size, image_size) * 0.3
        gray_matter = torch.randn(image_size, image_size) * 0.2
        
        # Create masks for different regions
        center_mask = torch.zeros(image_size, image_size)
        center_mask[image_size//4:3*image_size//4, image_size//4:3*image_size//4] = 1
        
        # Combine different tissue types
        image = background + center_mask * white_matter + (1 - center_mask) * gray_matter
        
        # Add some lesion-like structures (randomly)
        if torch.rand(1) < 0.3:
            lesion_mask = torch.zeros(image_size, image_size)
            lesion_x = torch.randint(image_size//4, 3*image_size//4, (1,))
            lesion_y = torch.randint(image_size//4, 3*image_size//4, (1,))
            lesion_mask[lesion_y-2:lesion_y+2, lesion_x-2:lesion_x+2] = 1
            image += lesion_mask * torch.randn(image_size, image_size) * 0.5
        
        nex1_images[i, 0] = image
    
    # Create NEX=6 images by averaging multiple NEX=1 acquisitions
    optimizer = NoiseScheduleOptimizer(NoiseScheduleConfig())
    nex6_images = optimizer.create_nex6_from_nex1(nex1_images)
    
    return nex1_images, nex6_images


def compare_noise_schedules(nex1_images: torch.Tensor, nex6_images: torch.Tensor):
    """
    Compare different noise schedules and their performance.
    
    Args:
        nex1_images: NEX=1 image data
        nex6_images: NEX=6 image data
    """
    print("Comparing different noise schedules...")
    
    # Create optimizer
    config = NoiseScheduleConfig(
        num_timesteps=1000,
        optimization_steps=200,
        learning_rate=1e-3
    )
    optimizer = NoiseScheduleOptimizer(config)
    
    # Test different schedule types
    schedule_types = ["linear", "cosine", "sigmoid"]
    results = {}
    
    for schedule_type in schedule_types:
        print(f"\nTesting {schedule_type} schedule...")
        
        # Create schedule
        schedule = optimizer.create_noise_schedule(schedule_type)
        
        # Evaluate schedule
        metrics = optimizer.evaluate_noise_schedule(nex6_images, nex1_images, schedule)
        
        results[schedule_type] = {
            "schedule": schedule,
            "metrics": metrics
        }
        
        print(f"  SNR Ratio: {metrics['snr_ratio']:.4f}")
        print(f"  Similarity: {metrics['similarity']:.4f}")
    
    # Optimize custom schedule
    print("\nOptimizing custom schedule...")
    optimized_schedule, history = optimizer.optimize_noise_schedule(nex6_images, nex1_images)
    
    # Evaluate optimized schedule
    optimized_metrics = optimizer.evaluate_noise_schedule(nex6_images, nex1_images, optimized_schedule)
    results["optimized"] = {
        "schedule": optimized_schedule,
        "metrics": optimized_metrics,
        "history": history
    }
    
    print(f"  Optimized SNR Ratio: {optimized_metrics['snr_ratio']:.4f}")
    print(f"  Optimized Similarity: {optimized_metrics['similarity']:.4f}")
    
    return results


def visualize_comparison(results: dict, save_path: str = None):
    """
    Visualize comparison of different noise schedules.
    
    Args:
        results: Results from compare_noise_schedules
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot beta schedules
    for i, (schedule_type, result) in enumerate(results.items()):
        row = i // 3
        col = i % 3
        
        schedule = result["schedule"]
        metrics = result["metrics"]
        
        # Plot beta values
        axes[row, col].plot(schedule.detach().cpu().numpy())
        axes[row, col].set_title(f"{schedule_type.capitalize()} Schedule\n"
                               f"SNR Ratio: {metrics['snr_ratio']:.3f}\n"
                               f"Similarity: {metrics['similarity']:.3f}")
        axes[row, col].set_xlabel("Timestep")
        axes[row, col].set_ylabel("Beta")
        axes[row, col].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def train_with_optimized_schedule(nex1_images: torch.Tensor, nex6_images: torch.Tensor):
    """
    Train a diffusion model with optimized noise schedule.
    
    Args:
        nex1_images: NEX=1 image data
        nex6_images: NEX=6 image data
    """
    print("\nTraining diffusion model with optimized noise schedule...")
    
    # Create model with optimized schedule
    config = NoiseScheduleConfig(
        num_timesteps=1000,
        target_snr_ratio=1.0,
        optimization_steps=100
    )
    
    model = DiffusionModel(
        use_optimized_schedule=True,
        schedule_config=config,
        adaptation_frequency=50
    )
    
    # Optimize schedule offline first
    print("Optimizing noise schedule offline...")
    optimization_results = model.optimize_noise_schedule_offline(nex1_images, nex6_images)
    
    print(f"Optimization completed!")
    print(f"Final SNR Ratio: {optimization_results['schedule_info']['mean_beta']:.4f}")
    
    # Create simple training loop (for demonstration)
    print("Training model...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Simple training loop
    for epoch in range(10):
        total_loss = 0
        num_batches = 0
        
        # Create batches
        batch_size = 16
        for i in range(0, len(nex1_images), batch_size):
            batch_images = nex1_images[i:i+batch_size]
            
            # Training step
            optimizer.zero_grad()
            
            # Add noise
            noise = torch.randn_like(batch_images)
            timesteps = torch.randint(0, model.scheduler.num_train_timesteps, 
                                    (batch_images.size(0),), device=batch_images.device)
            
            if model.use_optimized_schedule:
                noisy_images = model.scheduler.add_noise_with_schedule(batch_images, noise, timesteps)
            else:
                noisy_images = model.scheduler.add_noise(batch_images, noise, timesteps)
            
            # Predict noise
            predicted_noise = model.model(noisy_images, timesteps).sample
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
    
    return model, optimization_results


def main():
    """Main function demonstrating the complete workflow."""
    print("Medical Imaging Noise Schedule Optimization Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\n1. Creating synthetic medical imaging data...")
    nex1_images, nex6_images = create_synthetic_medical_data(num_samples=200, image_size=64)
    nex1_images = nex1_images.to(device)
    nex6_images = nex6_images.to(device)
    
    print(f"Created {len(nex1_images)} samples")
    print(f"NEX=1 images shape: {nex1_images.shape}")
    print(f"NEX=6 images shape: {nex6_images.shape}")
    
    # Compare different noise schedules
    print("\n2. Comparing different noise schedules...")
    results = compare_noise_schedules(nex1_images, nex6_images)
    
    # Visualize comparison
    print("\n3. Visualizing schedule comparison...")
    visualize_comparison(results, "noise_schedule_comparison.png")
    
    # Train with optimized schedule
    print("\n4. Training with optimized schedule...")
    model, optimization_results = train_with_optimized_schedule(nex1_images, nex6_images)
    
    # Final evaluation
    print("\n5. Final evaluation...")
    final_schedule = model.scheduler.betas
    optimizer = NoiseScheduleOptimizer(NoiseScheduleConfig())
    final_metrics = optimizer.evaluate_noise_schedule(nex6_images, nex1_images, final_schedule)
    
    print(f"Final optimized schedule performance:")
    print(f"  SNR Ratio: {final_metrics['snr_ratio']:.4f}")
    print(f"  Similarity: {final_metrics['similarity']:.4f}")
    
    # Visualize final schedule
    print("\n6. Visualizing final optimized schedule...")
    optimizer.visualize_noise_schedule(final_schedule, "final_optimized_schedule.png")
    
    print("\nDemo completed successfully!")
    print("Check the generated plots for visualization of the results.")


if __name__ == "__main__":
    main()
