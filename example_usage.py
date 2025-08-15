#!/usr/bin/env python3
"""
Example usage of DWI DataLoader with DDPM model

This script demonstrates how to use the filtered DWI dataset with your DDPM model.
"""

import torch
import numpy as np
from src.data.dwi_dataloader import create_dwi_dataloader
from src.model.ssddpm import SSDDPM


def main():
    """Example usage of the DWI dataloader with DDPM"""

    print("=== DWI DDPM Example Usage ===\n")

    # 1. Create the dataloader
    print("1. Creating DWI dataloader...")
    dataloader = create_dwi_dataloader(
        filtered_cases_file="filtered_cases.txt",
        batch_size=2,
        target_size=(64, 64, 64),  # Match your model's sample_size
        max_cases=5,  # Limit for testing
        num_workers=0,  # Use 0 for debugging
        shuffle=True,
    )

    print(f"   ✓ Dataloader created with {len(dataloader.dataset)} cases")

    # 2. Create the DDPM model
    print("\n2. Creating DDPM model...")
    model = SSDDPM(
        in_channels=1,  # Single channel for DWI
        out_channels=1,
        sample_size=64,  # Match target_size
        num_timesteps=1000,
    )

    print(
        f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # 3. Test the pipeline
    print("\n3. Testing the pipeline...")

    # Get a sample batch
    batch = next(iter(dataloader))
    x_0 = batch["volume"]  # Clean images
    batch_size = x_0.shape[0]

    print(f"   ✓ Loaded batch with shape: {x_0.shape}")
    print(f"   ✓ Case names: {batch['case_name']}")
    print(f"   ✓ BVAL values shape: {batch['bval_values'].shape}")

    # Test DDPM sampling
    timesteps = model.sample_timesteps(batch_size)
    epsilon = model.sample_noise(x_0.shape)

    print(f"   ✓ Sampled timesteps: {timesteps}")
    print(f"   ✓ Sampled noise shape: {epsilon.shape}")

    # Test forward pass
    with torch.no_grad():
        predicted_epsilon = model(x_0, timesteps)

    print(f"   ✓ Model forward pass successful!")
    print(f"   ✓ Predicted noise shape: {predicted_epsilon.sample.shape}")

    # 4. Show data statistics
    print("\n4. Data statistics:")
    print(f"   ✓ Volume range: [{x_0.min():.3f}, {x_0.max():.3f}]")
    print(f"   ✓ Volume mean: {x_0.mean():.3f}")
    print(f"   ✓ Volume std: {x_0.std():.3f}")

    # 5. Show bval information
    print("\n5. BVAL information:")
    bval_values = batch["bval_values"][0]  # First case
    print(f"   ✓ BVAL sequence: {bval_values[:10]}...")  # First 10 values
    print(f"   ✓ Number of diffusion directions: {len(bval_values)}")
    print(f"   ✓ Unique b-values: {np.unique(bval_values)}")

    print("\n=== Example completed successfully! ===")
    print("\nYou can now use this dataloader for training your DDPM model.")
    print("The dataloader provides:")
    print("- Clean DWI images (b0 images)")
    print("- Corresponding bval/bvec information")
    print("- Proper normalization and resizing")
    print("- Integration with DDPM timestep and noise sampling")


if __name__ == "__main__":
    main()
