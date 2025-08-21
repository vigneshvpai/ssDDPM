"""
PyTorch Dataset and DataLoader for 2D DWI data.
Uses the preprocessing module for data processing.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union

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


class DWIDataLoader:
    """
    Utility class for constructing 2D DWI dataloaders.
    """

    @staticmethod
    def create_dataloaders(
        train_data_list: str,
        val_data_list: str,
        test_data_list: str,
        batch_size: int = None,
        num_workers: int = None,
        **dataset_kwargs,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """
        Build train, validation, and test dataloaders for 2D DWI patches.

        Args:
            train_data_list: Path to the training data list JSON.
            val_data_list: Path to the validation data list JSON.
            test_data_list: Path to the test data list JSON.
            batch_size: Batch size (defaults to Config.BATCH_SIZE if None).
            num_workers: Number of worker processes (defaults to Config.NUM_WORKERS if None).
            **dataset_kwargs: Additional arguments for DWIDataset.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        batch_size = batch_size or Config.BATCH_SIZE
        num_workers = num_workers or Config.NUM_WORKERS

        # Set default dataset parameters for 2D patch extraction
        dataset_kwargs.setdefault("normalize_to_b0", Config.NORMALIZE_TO_B0)
        dataset_kwargs.setdefault("patch_size", Config.PATCH_SIZE)
        dataset_kwargs.setdefault("patch_overlap", Config.PATCH_OVERLAP)
        dataset_kwargs.setdefault("max_b_values", Config.MAX_B_VALUES)

        # Instantiate datasets
        train_dataset = DWIDataset(train_data_list, **dataset_kwargs)
        val_dataset = DWIDataset(val_data_list, **dataset_kwargs)
        test_dataset = DWIDataset(test_data_list, **dataset_kwargs)

        # Create PyTorch dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY,
        )

        return train_loader, val_loader, test_loader
