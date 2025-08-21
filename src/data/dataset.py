"""
2D-compatible PyTorch Dataset for DWI data processing.
Splits 3D volumes into 2D slices and creates 128x128 patches for 2D UNet.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Union
import warnings

from src.config import Config


class DWIDataset(Dataset):
    """
    2D-compatible PyTorch Dataset for DWI data.
    Splits 3D volumes into 2D slices and creates 128x128 patches.
    """

    def __init__(
        self,
        data_list_path: str,
        transform=None,
        normalize_to_b0: bool = True,
        patch_size: int = 128,
        patch_overlap: int = 32,
        max_b_values: int = 25,
    ):
        """
        Initialize 2D DWI dataset.

        Args:
            data_list_path: Path to JSON file containing data list
            transform: Optional transform to apply to the data
            normalize_to_b0: Whether to normalize with respect to first b-value
            patch_size: Size of patches (default: 128)
            patch_overlap: Overlap between patches (default: 32)
            max_b_values: Maximum number of b-values to use (default: 25)
        """
        self.transform = transform
        self.normalize_to_b0 = normalize_to_b0
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.max_b_values = max_b_values

        # Load data list
        with open(data_list_path, "r") as f:
            self.data_list = json.load(f)

        # Pre-compute all patches for faster loading
        self.patches = self._create_patches()

        print(f"Loaded {len(self.data_list)} DWI acquisitions from {data_list_path}")
        print(f"Created {len(self.patches)} 2D patches")

    def _create_patches(self) -> List[Dict]:
        """
        Create patches from all DWI acquisitions.

        Returns:
            List of patch dictionaries
        """
        patches = []

        for item in self.data_list:
            # Load DWI data
            dwi_data = self._load_dwi_data(item["nii_path"])

            # Limit to max_b_values
            if dwi_data.shape[0] > self.max_b_values:
                dwi_data = dwi_data[: self.max_b_values]

            # Split into 2D slices
            slices = self._split_into_slices(dwi_data)

            # Create patches from each slice
            for slice_idx, slice_data in enumerate(slices):
                slice_patches = self._create_slice_patches(slice_data, item, slice_idx)
                patches.extend(slice_patches)

        return patches

    def _load_dwi_data(self, nii_path: str) -> torch.Tensor:
        """
        Load DWI data from .nii.gz file.

        Args:
            nii_path: Path to .nii.gz file

        Returns:
            DWI data as torch tensor with shape (b_values, width, height, slices)
        """
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()

            # Convert to torch tensor
            data = torch.from_numpy(data).float()

            # Ensure correct shape: (width, height, slices, b_values) -> (b_values, width, height, slices)
            if len(data.shape) == 4:
                data = data.permute(3, 0, 1, 2)

            return data

        except Exception as e:
            raise RuntimeError(f"Error loading DWI data from {nii_path}: {e}")

    def _split_into_slices(self, dwi_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split 3D volume into 2D slices.

        Args:
            dwi_data: DWI data tensor of shape (b_values, width, height, slices)

        Returns:
            List of 2D slices, each of shape (b_values, width, height)
        """
        slices = []
        for slice_idx in range(dwi_data.shape[3]):  # Iterate over slices
            slice_data = dwi_data[:, :, :, slice_idx]  # (b_values, width, height)
            slices.append(slice_data)
        return slices

    def _create_slice_patches(
        self, slice_data: torch.Tensor, item: Dict, slice_idx: int
    ) -> List[Dict]:
        """
        Create patches from a single 2D slice.

        Args:
            slice_data: 2D slice tensor of shape (b_values, width, height)
            item: Original data item
            slice_idx: Index of the slice

        Returns:
            List of patch dictionaries
        """
        patches = []
        height, width = slice_data.shape[1], slice_data.shape[2]

        # Calculate patch positions
        stride = self.patch_size - self.patch_overlap

        for y in range(0, height - self.patch_size + 1, stride):
            for x in range(0, width - self.patch_size + 1, stride):
                # Extract patch
                patch = slice_data[:, y : y + self.patch_size, x : x + self.patch_size]

                # Pad if necessary to reach patch_size
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    pad_h = max(0, self.patch_size - patch.shape[1])
                    pad_w = max(0, self.patch_size - patch.shape[2])
                    patch = torch.nn.functional.pad(
                        patch, (0, pad_w, 0, pad_h), mode="reflect"
                    )

                # Normalize with respect to b0 if requested
                if self.normalize_to_b0:
                    patch = self._normalize_to_b0(patch)

                # Create patch info
                patch_info = {
                    "patch_data": patch,
                    "subject_id": item["subject_id"],
                    "acquisition_id": item["acquisition_id"],
                    "slice_idx": slice_idx,
                    "patch_x": x,
                    "patch_y": y,
                    "original_shape": item["shape"],
                    "b_values": item["b_values"][: self.max_b_values],
                    "b_vectors": (
                        item["b_vectors"][: self.max_b_values]
                        if len(item["b_vectors"]) >= self.max_b_values
                        else item["b_vectors"]
                    ),
                }

                patches.append(patch_info)

        return patches

    def _normalize_to_b0(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Normalize patch with respect to the first b-value (b0).

        Args:
            patch: Patch tensor of shape (b_values, height, width)

        Returns:
            Normalized patch
        """
        b0 = patch[0:1]  # First b-value (b0)

        # Avoid division by zero
        b0 = torch.clamp(b0, min=1e-8)

        # Normalize all b-values with respect to b0
        normalized_patch = patch / b0

        # Clip to reasonable range [0, 1] for most DWI data
        normalized_patch = torch.clamp(normalized_patch, 0, 1)

        return normalized_patch

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single 2D patch.

        Args:
            idx: Index of the patch

        Returns:
            Dictionary containing patch data and metadata
        """
        patch_info = self.patches[idx]

        # Get patch data
        patch_data = patch_info["patch_data"]

        # Apply transforms if provided
        if self.transform:
            patch_data = self.transform(patch_data)

        return {
            "dwi": patch_data,  # Shape: (b_values, 128, 128)
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
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        subjects = set(patch["subject_id"] for patch in self.patches)
        acquisitions = set(patch["acquisition_id"] for patch in self.patches)

        return {
            "num_patches": len(self.patches),
            "num_subjects": len(subjects),
            "num_acquisitions": len(acquisitions),
            "patch_size": self.patch_size,
            "patch_overlap": self.patch_overlap,
            "max_b_values": self.max_b_values,
            "subjects": list(subjects),
        }


class DWIDataLoader:
    """
    Convenience class for creating 2D DWI dataloaders.
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
        Create train, validation, and test dataloaders for 2D patches.

        Args:
            train_data_list: Path to training data list
            val_data_list: Path to validation data list
            test_data_list: Path to test data list
            batch_size: Batch size (uses Config.BATCH_SIZE if None)
            num_workers: Number of workers (uses Config.NUM_WORKERS if None)
            **dataset_kwargs: Additional arguments for DWIDataset2D

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or Config.BATCH_SIZE
        num_workers = num_workers or Config.NUM_WORKERS

        # Use 2D-specific defaults
        dataset_kwargs.setdefault("normalize_to_b0", Config.NORMALIZE_TO_B0)
        dataset_kwargs.setdefault("patch_size", Config.PATCH_SIZE)
        dataset_kwargs.setdefault("patch_overlap", Config.PATCH_OVERLAP)
        dataset_kwargs.setdefault("max_b_values", Config.MAX_B_VALUES)

        # Create datasets
        train_dataset = DWIDataset(train_data_list, **dataset_kwargs)
        val_dataset = DWIDataset(val_data_list, **dataset_kwargs)
        test_dataset = DWIDataset(test_data_list, **dataset_kwargs)

        # Create dataloaders
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
