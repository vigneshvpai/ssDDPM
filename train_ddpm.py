#!/usr/bin/env python3
"""
DDPM Training Script for DWI Data

This script demonstrates how to train the DDPM model using the filtered DWI dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Import our modules
from src.model.ssddpm import SSDDPM
from src.data.dwi_dataloader import create_dwi_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_noise_to_image(x_0, epsilon, timesteps, num_timesteps=1000):
    """
    Add noise to images based on timesteps (simplified DDPM noise schedule)
    
    Args:
        x_0: Clean images (batch_size, channels, height, width)
        epsilon: Noise tensor (batch_size, channels, height, width)
        timesteps: Timestep tensor (batch_size,)
        num_timesteps: Total number of timesteps
        
    Returns:
        Noisy images
    """
    # Simple linear noise schedule
    beta_t = timesteps.float() / num_timesteps
    
    # Expand dimensions for broadcasting
    beta_t = beta_t.view(-1, 1, 1, 1)
    
    # Add noise: x_t = sqrt(1 - beta_t) * x_0 + sqrt(beta_t) * epsilon
    x_t = torch.sqrt(1 - beta_t) * x_0 + torch.sqrt(beta_t) * epsilon
    
    return x_t


def train_ddpm(
    model,
    dataloader,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='checkpoints'
):
    """
    Train the DDPM model
    
    Args:
        model: SSDDPM model
        dataloader: DWI dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save checkpoints
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    logger.info(f"Starting training on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get clean images
            x_0 = batch['volume'].to(device)  # (batch_size, channels, height, width)
            batch_size = x_0.shape[0]
            
            # Sample timesteps
            timesteps = model.sample_timesteps(batch_size, device=device)
            
            # Sample noise
            epsilon = model.sample_noise(x_0.shape, device=device)
            
            # Add noise to images
            x_t = add_noise_to_image(x_0, epsilon, timesteps, model.num_timesteps)
            
            # Predict noise
            predicted_epsilon = model(x_t, timesteps)
            
            # Compute loss
            loss = criterion(predicted_epsilon, epsilon)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/num_batches:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_path / f"ddpm_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training function"""
    # Configuration
    config = {
        'filtered_cases_file': 'filtered_cases.txt',
        'batch_size': 4,
        'target_size': (64, 64, 64),  # Match your model's sample_size
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'max_cases': None,  # Set to a small number for testing
        'num_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("Initializing DDPM training...")
    logger.info(f"Configuration: {config}")
    
    # Check if filtered cases file exists
    if not Path(config['filtered_cases_file']).exists():
        logger.error(f"Filtered cases file not found: {config['filtered_cases_file']}")
        logger.error("Please run the filtering script first.")
        return
    
    # Create dataloader
    logger.info("Creating DWI dataloader...")
    dataloader = create_dwi_dataloader(
        filtered_cases_file=config['filtered_cases_file'],
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        max_cases=config['max_cases'],
        num_workers=config['num_workers'],
        shuffle=True
    )
    
    logger.info(f"Dataloader created with {len(dataloader.dataset)} cases")
    
    # Create model
    logger.info("Creating DDPM model...")
    model = SSDDPM(
        in_channels=1,  # Single channel for DWI
        out_channels=1,
        sample_size=config['target_size'][0],  # Use first dimension
        num_timesteps=1000
    )
    
    # Test the model with a sample batch
    logger.info("Testing model with sample batch...")
    sample_batch = next(iter(dataloader))
    x_0 = sample_batch['volume']
    batch_size = x_0.shape[0]
    
    # Test forward pass
    timesteps = model.sample_timesteps(batch_size, device=config['device'])
    epsilon = model.sample_noise(x_0.shape, device=config['device'])
    x_t = add_noise_to_image(x_0, epsilon, timesteps, model.num_timesteps)
    
    with torch.no_grad():
        predicted_epsilon = model(x_t, timesteps)
    
    logger.info(f"Model test successful!")
    logger.info(f"Input shape: {x_0.shape}")
    logger.info(f"Output shape: {predicted_epsilon.shape}")
    
    # Start training
    logger.info("Starting training...")
    train_ddpm(
        model=model,
        dataloader=dataloader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        save_path='checkpoints'
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
