"""
PyTorch Dataset class for DWI data loading.
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
    PyTorch Dataset for DWI data with .nii.gz, .bval, and .bvec files.
    """

    def __init__(
        self,
        data_list_path: str,
        transform=None,
        normalize: bool = True,
        target_shape: Optional[Tuple] = None,
    ):
        """
        Initialize DWI dataset.

        Args:
            data_list_path: Path to JSON file containing data list
            transform: Optional transform to apply to the data
            normalize: Whether to normalize the DWI data
            target_shape: Target shape for resizing (width, height, slices)
        """
        self.transform = transform
        self.normalize = normalize
        self.target_shape = target_shape

        # Load data list
        with open(data_list_path, "r") as f:
            self.data_list = json.load(f)

        print(f"Loaded {len(self.data_list)} DWI acquisitions from {data_list_path}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single DWI acquisition.

        Args:
            idx: Index of the data item

        Returns:
            Dictionary containing DWI data, b-values, b-vectors, and metadata
        """
        item = self.data_list[idx]

        # Load DWI data
        dwi_data = self._load_dwi_data(item["nii_path"])

        # Load b-values and b-vectors
        bvals = torch.tensor(item["b_values"], dtype=torch.float32)
        bvecs = torch.tensor(item["b_vectors"], dtype=torch.float32)

        # Preprocess DWI data
        if self.normalize:
            dwi_data = self._normalize_dwi(dwi_data)

        if self.target_shape:
            dwi_data = self._resize_dwi(dwi_data, self.target_shape)

        # Apply transforms if provided
        if self.transform:
            dwi_data = self.transform(dwi_data)

        return {
            "dwi": dwi_data,
            "bvals": bvals,
            "bvecs": bvecs,
            "subject_id": item["subject_id"],
            "acquisition_id": item["acquisition_id"],
            "original_shape": item["shape"],
        }

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

    def _normalize_dwi(self, dwi_data: torch.Tensor) -> torch.Tensor:
        """
        Normalize DWI data.

        Args:
            dwi_data: DWI data tensor

        Returns:
            Normalized DWI data
        """
        # Normalize each b-value volume independently
        normalized_data = torch.zeros_like(dwi_data)

        for i in range(dwi_data.shape[0]):
            volume = dwi_data[i]

            # Remove outliers (top and bottom 1%)
            flat_volume = volume.flatten()
            sorted_values, _ = torch.sort(flat_volume)
            n = len(sorted_values)
            lower_percentile = sorted_values[int(0.01 * n)]
            upper_percentile = sorted_values[int(0.99 * n)]

            # Clip outliers
            volume = torch.clamp(volume, lower_percentile, upper_percentile)

            # Normalize to [0, 1]
            if volume.max() > volume.min():
                volume = (volume - volume.min()) / (volume.max() - volume.min())

            normalized_data[i] = volume

        return normalized_data

    def _resize_dwi(self, dwi_data: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """
        Resize DWI data to target shape using interpolation.

        Args:
            dwi_data: DWI data tensor
            target_shape: Target shape (width, height, slices)

        Returns:
            Resized DWI data
        """
        import torch.nn.functional as F

        current_shape = dwi_data.shape[1:]  # (b_values, width, height, slices)

        if current_shape == target_shape:
            return dwi_data

        # Resize each b-value volume
        resized_data = torch.zeros(
            dwi_data.shape[0], *target_shape, device=dwi_data.device
        )

        for i in range(dwi_data.shape[0]):
            volume = dwi_data[i]  # (width, height, slices)

            # Add batch and channel dimensions for interpolation
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, width, height, slices)

            # Resize using trilinear interpolation
            resized_volume = F.interpolate(
                volume, size=target_shape, mode="trilinear", align_corners=False
            )

            resized_data[i] = resized_volume.squeeze(0).squeeze(0)

        return resized_data

    def get_data_info(self) -> Dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        shapes = [
            tuple(item["shape"]) for item in self.data_list
        ]  # Convert to tuples for hashing
        b_values = [item["num_b_values"] for item in self.data_list]
        subjects = set(item["subject_id"] for item in self.data_list)

        return {
            "num_samples": len(self.data_list),
            "num_subjects": len(subjects),
            "unique_shapes": list(set(shapes)),
            "unique_b_values": list(set(b_values)),
            "subjects": list(subjects),
        }


class DWIDataLoader:
    """
    Convenience class for creating DWI dataloaders.
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
        Create train, validation, and test dataloaders.

        Args:
            train_data_list: Path to training data list
            val_data_list: Path to validation data list
            test_data_list: Path to test data list
            batch_size: Batch size (uses Config.BATCH_SIZE if None)
            num_workers: Number of workers (uses Config.NUM_WORKERS if None)
            **dataset_kwargs: Additional arguments for DWIDataset

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or Config.BATCH_SIZE
        num_workers = num_workers or Config.NUM_WORKERS

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
