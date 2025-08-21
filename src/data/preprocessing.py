"""
DWI data preprocessing utilities.
Handles loading NIfTI files, creating patches, and normalization.
"""

import numpy as np
import torch
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Union
import warnings

from src.config import Config


class DWIPreprocessor:
    """
    Utility class for preprocessing DWI data.
    Handles loading, slicing, patching, and normalization.
    """

    def __init__(
        self,
        normalize_to_b0: bool = True,
        patch_size: int = 64,
        patch_overlap: int = 16,
        max_b_values: int = 25,
    ):
        """
        Initialize the DWI preprocessor.

        Args:
            normalize_to_b0: If True, normalize each patch by the first b-value (b0).
            patch_size: Size of each patch (default: 64).
            patch_overlap: Overlap between adjacent patches (default: 16).
            max_b_values: Maximum number of b-value channels to use (default: 25).
        """
        self.normalize_to_b0 = normalize_to_b0
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.max_b_values = max_b_values

    def load_dwi_data(self, nii_path: str) -> torch.Tensor:
        """
        Load DWI data from a NIfTI (.nii.gz) file.

        Args:
            nii_path: Path to the NIfTI file.

        Returns:
            DWI data as a torch tensor with shape (b_values, width, height, slices).
        """
        try:
            img = nib.load(nii_path)
            data = img.get_fdata()

            # Convert numpy array to torch tensor
            data = torch.from_numpy(data).float()

            # Rearrange axes if necessary: (width, height, slices, b_values) -> (b_values, width, height, slices)
            if len(data.shape) == 4:
                data = data.permute(3, 0, 1, 2)

            return data

        except Exception as e:
            raise RuntimeError(f"Error loading DWI data from {nii_path}: {e}")

    def split_into_slices(self, dwi_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split a 3D DWI volume into a list of 2D slices.

        Args:
            dwi_data: DWI tensor of shape (b_values, width, height, slices).

        Returns:
            List of 2D slice tensors, each of shape (b_values, width, height).
        """
        slices = []
        for slice_idx in range(dwi_data.shape[3]):  # Loop over slices
            slice_data = dwi_data[:, :, :, slice_idx]  # (b_values, width, height)
            slices.append(slice_data)
        return slices

    def create_slice_patches(
        self, slice_data: torch.Tensor, item: Dict, slice_idx: int
    ) -> List[Dict]:
        """
        Generate patches from a single 2D slice.

        Args:
            slice_data: 2D slice tensor of shape (b_values, width, height).
            item: Dictionary with metadata for the current acquisition.
            slice_idx: Index of the current slice.

        Returns:
            List of dictionaries, each describing a patch.
        """
        patches = []
        width, height = slice_data.shape[1], slice_data.shape[2]

        # Compute the stride for patch extraction
        stride = self.patch_size - self.patch_overlap

        for y in range(0, height - self.patch_size + 1, stride):
            for x in range(0, width - self.patch_size + 1, stride):
                # Extract the patch
                patch = slice_data[:, x : x + self.patch_size, y : y + self.patch_size]

                # Pad the patch if it is smaller than patch_size
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    pad_h = max(0, self.patch_size - patch.shape[1])
                    pad_w = max(0, self.patch_size - patch.shape[2])
                    patch = torch.nn.functional.pad(
                        patch, (0, pad_w, 0, pad_h), mode="reflect"
                    )

                # Optionally normalize the patch by b0
                if self.normalize_to_b0:
                    patch = self._normalize_to_b0(patch)

                # Store patch and metadata
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
        Normalize a patch by the first b-value (b0).

        Args:
            patch: Patch tensor of shape (b_values, height, width).

        Returns:
            Patch normalized by b0.
        """
        b0 = patch[0:1]  # First b-value (b0)

        # Prevent division by zero
        b0 = torch.clamp(b0, min=1e-8)

        # Normalize all b-values by b0
        normalized_patch = patch / b0

        # Clamp values to [0, 1] for typical DWI data
        normalized_patch = torch.clamp(normalized_patch, 0, 1)

        return normalized_patch

    def create_patches_from_data_list(self, data_list: List[Dict]) -> List[Dict]:
        """
        Generate all patches from a list of DWI acquisitions.

        Args:
            data_list: List of dictionaries, each describing a DWI acquisition.

        Returns:
            List of dictionaries, each describing a patch.
        """
        patches = []

        for item in data_list:
            # Load the DWI volume
            dwi_data = self.load_dwi_data(item["nii_path"])

            # Restrict to the maximum number of b-values
            if dwi_data.shape[0] > self.max_b_values:
                dwi_data = dwi_data[: self.max_b_values]

            # Split the volume into 2D slices
            slices = self.split_into_slices(dwi_data)

            # Generate patches for each slice
            for slice_idx, slice_data in enumerate(slices):
                slice_patches = self.create_slice_patches(slice_data, item, slice_idx)
                patches.extend(slice_patches)

        return patches
