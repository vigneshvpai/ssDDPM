#!/usr/bin/env python3
"""
Dataset summary script for DWI data.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import DWIDataset
import json


def main():
    """Print comprehensive dataset summary."""

    print("=== DWI Dataset Summary ===\n")

    # Load all datasets
    train_dataset = DWIDataset(
        str(Config.TRAIN_DATA_LIST),
        normalize_to_b0=True,
        patch_size=128,
        patch_overlap=32,
        max_b_values=25,
    )
    val_dataset = DWIDataset(
        str(Config.VAL_DATA_LIST),
        normalize_to_b0=True,
        patch_size=128,
        patch_overlap=32,
        max_b_values=25,
    )
    test_dataset = DWIDataset(
        str(Config.TEST_DATA_LIST),
        normalize_to_b0=True,
        patch_size=128,
        patch_overlap=32,
        max_b_values=25,
    )

    # Get info
    train_info = train_dataset.get_data_info()
    val_info = val_dataset.get_data_info()
    test_info = test_dataset.get_data_info()

    print("Dataset Splits:")
    print(
        f"  Training:   {train_info['num_patches']:3d} patches from {train_info['num_subjects']:3d} subjects"
    )
    print(
        f"  Validation: {val_info['num_patches']:3d} patches from {val_info['num_subjects']:3d} subjects"
    )
    print(
        f"  Test:       {test_info['num_patches']:3d} patches from {test_info['num_subjects']:3d} subjects"
    )
    print(
        f"  Total:      {train_info['num_patches'] + val_info['num_patches'] + test_info['num_patches']:3d} patches from {len(set(train_info['subjects'] + val_info['subjects'] + test_info['subjects'])):3d} subjects"
    )

    print(f"\nPatch Information:")
    print(f"  Patch size: {train_info['patch_size']}x{train_info['patch_size']}")
    print(f"  Patch overlap: {train_info['patch_overlap']}")
    print(f"  Max b-values: {train_info['max_b_values']}")

    # Show sample data
    print(f"\nSample Data Structure:")
    sample = train_dataset[0]
    print(f"  DWI shape: {sample['dwi'].shape}")
    print(f"  B-values: {sample['bvals'][:10].tolist()}... (showing first 10)")
    print(f"  B-vectors shape: {sample['bvecs'].shape}")
    print(f"  Subject ID: {sample['subject_id']}")
    print(f"  Slice index: {sample['slice_idx']}")
    print(f"  Patch position: ({sample['patch_x']}, {sample['patch_y']})")
    print(f"  Original shape: {sample['original_shape']}")

    # Show preprocessing info
    print(f"\nPreprocessing:")
    print(f"  Patch size: {Config.PATCH_SIZE}")
    print(f"  Patch overlap: {Config.PATCH_OVERLAP}")
    print(f"  B0 normalization: {Config.NORMALIZE_TO_B0}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Number of workers: {Config.NUM_WORKERS}")

    # Show data loading performance
    print(f"\nData Loading Performance:")
    print(f"  Pin memory: {Config.PIN_MEMORY}")
    print(f"  Cache directory: {Config.CACHE_DIR}")
    print(f"  Data root: {Config.DATA_ROOT}")


if __name__ == "__main__":
    main()
