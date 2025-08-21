"""
Configuration file for DWI dataset and training parameters.
"""

import os
from pathlib import Path


class Config:
    # Dataset paths
    DATA_ROOT = os.environ.get("HPCVAULT", "/path/to/hpcvault") + "/data"

    # Output paths for pre-parsed data
    CACHE_DIR = Path("src/data/cache")
    TRAIN_DATA_LIST = CACHE_DIR / "train_data.json"
    VAL_DATA_LIST = CACHE_DIR / "val_data.json"
    TEST_DATA_LIST = CACHE_DIR / "test_data.json"

    # Data loading parameters
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # DWI specific parameters
    DWI_EXTENSIONS = [".nii.gz"]
    BVEC_EXTENSIONS = [".bvec"]
    BVAL_EXTENSIONS = [".bval"]

    # Data splitting
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Data preprocessing
    NORMALIZE_DWI = True
    CROP_TO_BRAIN = True
    TARGET_SHAPE = (96, 96, 64)  # (width, height, slices)

    # 2D UNet specific parameters
    PATCH_SIZE = 128  # 128x128 patches as per paper
    PATCH_OVERLAP = 32  # Overlap between patches
    NORMALIZE_TO_B0 = True  # Normalize with respect to first b-value
    MAX_B_VALUES = 25  # Use 25 b-value channels as per paper

    # Target b-value sequence for filtering
    TARGET_B_VALUES = [
        0,
        10,
        10,
        10,
        50,
        50,
        50,
        80,
        80,
        80,
        200,
        200,
        200,
        400,
        400,
        400,
        600,
        600,
        600,
        800,
        800,
        800,
        1000,
        1000,
        1000,
    ]

    # Model parameters
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
