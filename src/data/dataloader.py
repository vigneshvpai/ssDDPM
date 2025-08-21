"""
PyTorch Dataset and DataLoader for 2D DWI data.
Uses the preprocessing module for data processing.
Includes Lightning DataModule integration.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union

import lightning as L

from src.config import Config
from src.data.preprocessing import DWIPreprocessor


class DWIDataset(Dataset):
    """
    PyTorch Dataset for 2D DWI data.
    Converts 3D DWI volumes into 2D slices and generates patches.
    """

    def __init__(
        self,
        data_list_path: str,
        transform=None,
        normalize_to_b0: bool = True,
        patch_size: int = 64,
        patch_overlap: int = 16,
        max_b_values: int = 25,
    ):
        """
        Initialize the 2D DWI dataset.

        Args:
            data_list_path: Path to the JSON file containing the data list.
            transform: Optional transform to apply to each patch.
            normalize_to_b0: If True, normalize each patch by the first b-value (b0).
            patch_size: Size of each patch (default: 64).
            patch_overlap: Overlap between adjacent patches (default: 16).
            max_b_values: Maximum number of b-value channels to use (default: 25).
        """
        self.transform = transform

        # Initialize the preprocessor
        self.preprocessor = DWIPreprocessor(
            normalize_to_b0=normalize_to_b0,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            max_b_values=max_b_values,
        )

        # Load the data list from JSON
        with open(data_list_path, "r") as f:
            self.data_list = json.load(f)

        # Precompute all patches for efficient access
        self.patches = self.preprocessor.create_patches_from_data_list(self.data_list)

        print(f"Loaded {len(self.data_list)} DWI acquisitions from {data_list_path}")
        print(f"Created {len(self.patches)} 2D patches")

    def __len__(self) -> int:
        # Return the total number of patches
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single 2D patch and its metadata.

        Args:
            idx: Index of the patch.

        Returns:
            Dictionary with patch data and associated metadata.
        """
        patch_info = self.patches[idx]

        # Extract patch data
        patch_data = patch_info["patch_data"]

        # Apply optional transform
        if self.transform:
            patch_data = self.transform(patch_data)

        return {
            "dwi": patch_data,  # Shape: (b_values, patch_size, patch_size)
            "bvals": torch.tensor(patch_info["b_values"], dtype=torch.float32),
            "bvecs": torch.tensor(patch_info["b_vectors"], dtype=torch.float32),
            "subject_id": patch_info["subject_id"],
            "acquisition_id": patch_info["acquisition_id"],
            "slice_idx": patch_info["slice_idx"],
            "patch_x": patch_info["patch_x"],
            "patch_y": patch_info["patch_y"],
            "original_shape": patch_info["original_shape"],
        }

    def get_data_info(self) -> Dict:
        """
        Return summary information about the dataset.

        Returns:
            Dictionary with dataset statistics and configuration.
        """
        subjects = set(patch["subject_id"] for patch in self.patches)
        acquisitions = set(patch["acquisition_id"] for patch in self.patches)

        return {
            "num_patches": len(self.patches),
            "num_subjects": len(subjects),
            "num_acquisitions": len(acquisitions),
            "patch_size": self.preprocessor.patch_size,
            "patch_overlap": self.preprocessor.patch_overlap,
            "max_b_values": self.preprocessor.max_b_values,
            "subjects": list(subjects),
        }

    def analyze_patch_coverage(self) -> Dict:
        """
        Analyze patch coverage and distribution across different image sizes.

        Returns:
            Dictionary with coverage statistics.
        """
        coverage_stats = {}

        # Group patches by subject and acquisition
        subject_acq_patches = {}
        for patch in self.patches:
            key = (patch["subject_id"], patch["acquisition_id"])
            if key not in subject_acq_patches:
                subject_acq_patches[key] = []
            subject_acq_patches[key].append(patch)

        # Analyze each subject-acquisition combination
        for (subject_id, acq_id), patches in subject_acq_patches.items():
            # Group by slice
            slice_patches = {}
            for patch in patches:
                slice_idx = patch["slice_idx"]
                if slice_idx not in slice_patches:
                    slice_patches[slice_idx] = []
                slice_patches[slice_idx].append(patch)

            # Analyze each slice
            for slice_idx, slice_patch_list in slice_patches.items():
                original_shape = slice_patch_list[0]["original_shape"]
                width, height = (
                    original_shape[1],
                    original_shape[2],
                )  # Assuming shape is (b_values, width, height, slices)

                # Calculate expected number of patches
                stride = self.preprocessor.patch_size - self.preprocessor.patch_overlap
                expected_patches_x = max(1, (width + stride - 1) // stride)
                expected_patches_y = max(1, (height + stride - 1) // stride)
                expected_total = expected_patches_x * expected_patches_y

                # Get actual number of patches
                actual_total = len(slice_patch_list)

                # Calculate coverage percentage
                coverage_percentage = (
                    (actual_total / expected_total) * 100 if expected_total > 0 else 0
                )

                key = f"{subject_id}_{acq_id}_slice_{slice_idx}"
                coverage_stats[key] = {
                    "original_shape": original_shape,
                    "width": width,
                    "height": height,
                    "expected_patches_x": expected_patches_x,
                    "expected_patches_y": expected_patches_y,
                    "expected_total": expected_total,
                    "actual_total": actual_total,
                    "coverage_percentage": coverage_percentage,
                    "patch_size": self.preprocessor.patch_size,
                    "patch_overlap": self.preprocessor.patch_overlap,
                    "stride": stride,
                }

        return coverage_stats


class DWIDataLoader(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for 2D DWI data.

    This DataModule wraps the existing DWIDataset and provides a standardized
    interface for Lightning training, validation, and testing.
    """

    def __init__(
        self,
        train_data_list: str = None,
        val_data_list: str = None,
        test_data_list: str = None,
        batch_size: int = None,
        num_workers: int = None,
        normalize_to_b0: bool = None,
        patch_size: int = None,
        patch_overlap: int = None,
        max_b_values: int = None,
        pin_memory: bool = None,
    ):
        """
        Initialize the DWI Lightning DataModule.

        Args:
            train_data_list: Path to training data list JSON (defaults to Config.TRAIN_DATA_LIST)
            val_data_list: Path to validation data list JSON (defaults to Config.VAL_DATA_LIST)
            test_data_list: Path to test data list JSON (defaults to Config.TEST_DATA_LIST)
            batch_size: Batch size for all dataloaders (defaults to Config.BATCH_SIZE)
            num_workers: Number of worker processes (defaults to Config.NUM_WORKERS)
            normalize_to_b0: Normalize patches by b0 (defaults to Config.NORMALIZE_TO_B0)
            patch_size: Size of patches (defaults to Config.PATCH_SIZE)
            patch_overlap: Overlap between patches (defaults to Config.PATCH_OVERLAP)
            max_b_values: Maximum b-value channels (defaults to Config.MAX_B_VALUES)
            pin_memory: Enable pin memory for GPU training (defaults to Config.PIN_MEMORY)
        """
        super().__init__()

        # Set default values from config
        self.train_data_list = train_data_list or str(Config.TRAIN_DATA_LIST)
        self.val_data_list = val_data_list or str(Config.VAL_DATA_LIST)
        self.test_data_list = test_data_list or str(Config.TEST_DATA_LIST)
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.num_workers = num_workers or Config.NUM_WORKERS
        self.normalize_to_b0 = (
            normalize_to_b0 if normalize_to_b0 is not None else Config.NORMALIZE_TO_B0
        )
        self.patch_size = patch_size or Config.PATCH_SIZE
        self.patch_overlap = patch_overlap or Config.PATCH_OVERLAP
        self.max_b_values = max_b_values or Config.MAX_B_VALUES
        self.pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY

        # Dataset instances (will be created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare data if needed.
        This is called once per node when using distributed training.
        """
        # Data is already prepared in JSON format, so we just verify files exist
        import os

        for data_list in [
            self.train_data_list,
            self.val_data_list,
            self.test_data_list,
        ]:
            if not os.path.exists(data_list):
                raise FileNotFoundError(f"Data list file not found: {data_list}")

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Define dataset parameters
        dataset_kwargs = {
            "normalize_to_b0": self.normalize_to_b0,
            "patch_size": self.patch_size,
            "patch_overlap": self.patch_overlap,
            "max_b_values": self.max_b_values,
        }

        # Create datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = DWIDataset(self.train_data_list, **dataset_kwargs)
            self.val_dataset = DWIDataset(self.val_data_list, **dataset_kwargs)

        if stage == "validate" or stage is None:
            if self.val_dataset is None:
                self.val_dataset = DWIDataset(self.val_data_list, **dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = DWIDataset(self.test_data_list, **dataset_kwargs)

        if stage == "predict" or stage is None:
            if self.test_dataset is None:
                self.test_dataset = DWIDataset(self.test_data_list, **dataset_kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            self.setup(stage="fit")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage="validate")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            self.setup(stage="test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        if self.test_dataset is None:
            self.setup(stage="predict")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_data_info(self) -> dict:
        """
        Get information about the datasets.

        Returns:
            Dictionary with dataset statistics and configuration.
        """
        if self.train_dataset is None:
            self.setup(stage="fit")

        train_info = self.train_dataset.get_data_info()
        val_info = self.val_dataset.get_data_info()
        test_info = self.test_dataset.get_data_info()

        return {
            "train": train_info,
            "validation": val_info,
            "test": test_info,
            "config": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "normalize_to_b0": self.normalize_to_b0,
                "patch_size": self.patch_size,
                "patch_overlap": self.patch_overlap,
                "max_b_values": self.max_b_values,
                "pin_memory": self.pin_memory,
            },
        }

    def analyze_patch_coverage(self) -> dict:
        """
        Analyze patch coverage for all datasets.

        Returns:
            Dictionary with coverage statistics for train, validation, and test sets.
        """
        if self.train_dataset is None:
            self.setup(stage="fit")

        return {
            "train": self.train_dataset.analyze_patch_coverage(),
            "validation": self.val_dataset.analyze_patch_coverage(),
            "test": self.test_dataset.analyze_patch_coverage(),
        }

    def get_sample_batch(self, stage: str = "train") -> dict:
        """
        Get a sample batch from the specified stage.

        Args:
            stage: Stage to get sample from ('train', 'val', 'test')

        Returns:
            Sample batch dictionary
        """
        if stage == "train":
            if self.train_dataset is None:
                self.setup(stage="fit")
            return self.train_dataset[0]
        elif stage == "val":
            if self.val_dataset is None:
                self.setup(stage="validate")
            return self.val_dataset[0]
        elif stage == "test":
            if self.test_dataset is None:
                self.setup(stage="test")
            return self.test_dataset[0]
        else:
            raise ValueError(
                f"Invalid stage: {stage}. Must be 'train', 'val', or 'test'"
            )
