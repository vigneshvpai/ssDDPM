#!/usr/bin/env python3
"""
Example usage of the DWI dataloader system.

This script demonstrates how to:
1. Pre-parse the dataset and create data lists
2. Create dataloaders for training
3. Iterate through the data
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from src.config import Config
from src.data.preprocessing import create_dataset_lists
from src.data.dataset import DWIDataLoader


def main():
    """Main function demonstrating dataloader usage."""

    print("=== DWI Dataloader Example ===\n")

    # Step 1: Create dataset lists (only need to run once)
    print("1. Creating dataset lists...")
    create_dataset_lists()
    print("Dataset lists created successfully!\n")

    # Step 2: Create dataloaders
    print("2. Creating dataloaders...")
    train_loader, val_loader, test_loader = DWIDataLoader.create_dataloaders(
        train_data_list=str(Config.TRAIN_DATA_LIST),
        val_data_list=str(Config.VAL_DATA_LIST),
        test_data_list=str(Config.TEST_DATA_LIST),
        normalize_to_b0=Config.NORMALIZE_TO_B0,
        patch_size=Config.PATCH_SIZE,
        patch_overlap=Config.PATCH_OVERLAP,
        max_b_values=Config.MAX_B_VALUES,
    )
    print("Dataloaders created successfully!\n")

    # Step 3: Example iteration through training data
    print("3. Example iteration through training data...")
    print(f"Number of training batches: {len(train_loader)}")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:  # Only show first 2 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  DWI shape: {batch['dwi'].shape}")
        print(f"  B-values shape: {batch['bvals'].shape}")
        print(f"  B-vectors shape: {batch['bvecs'].shape}")
        print(f"  Subject IDs: {batch['subject_id']}")
        print(f"  Acquisition IDs: {batch['acquisition_id']}")

        # Example: Print some statistics
        dwi_data = batch["dwi"]
        print(f"  DWI min/max: {dwi_data.min():.4f}/{dwi_data.max():.4f}")
        print(f"  DWI mean/std: {dwi_data.mean():.4f}/{dwi_data.std():.4f}")

    print("\n=== Example completed successfully! ===")


def test_single_sample():
    """Test loading a single sample from the dataset."""
    print("\n=== Testing Single Sample Loading ===")

    from src.data.dataset import DWIDataset

    # Create a single dataset
    dataset = DWIDataset(
        data_list_path=str(Config.TRAIN_DATA_LIST),
        normalize_to_b0=True,
        patch_size=128,
        patch_overlap=32,
        max_b_values=25,
    )

    print(f"Dataset size: {len(dataset)}")

    # Get dataset info
    info = dataset.get_data_info()
    print(f"Dataset info: {info}")

    # Load a single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"DWI shape: {sample['dwi'].shape}")
    print(f"B-values: {sample['bvals']}")
    print(f"Subject ID: {sample['subject_id']}")


if __name__ == "__main__":
    main()
    test_single_sample()
