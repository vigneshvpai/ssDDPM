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
        Generate patches from a single 2D slice with complete coverage.

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

        # Calculate the number of patches needed to cover the entire image
        # We want to ensure complete coverage, so we may need to extend beyond the image boundaries
        num_patches_x = max(1, (width + stride - 1) // stride)
        num_patches_y = max(1, (height + stride - 1) // stride)

        for patch_y_idx in range(num_patches_y):
            for patch_x_idx in range(num_patches_x):
                # Calculate the starting position for this patch
                x_start = patch_x_idx * stride
                y_start = patch_y_idx * stride

                # Calculate the ending position
                x_end = x_start + self.patch_size
                y_end = y_start + self.patch_size

                # Extract the patch from the image
                x_start_img = min(x_start, width)
                y_start_img = min(y_start, height)
                x_end_img = min(x_end, width)
                y_end_img = min(y_end, height)

                # Extract the valid portion of the patch
                patch = slice_data[:, x_start_img:x_end_img, y_start_img:y_end_img]

                # Create a full-size patch tensor
                full_patch = torch.zeros(
                    (slice_data.shape[0], self.patch_size, self.patch_size),
                    dtype=slice_data.dtype,
                    device=slice_data.device,
                )

                # Calculate the offset within the full patch
                x_offset = max(0, x_start_img - x_start)
                y_offset = max(0, y_start_img - y_start)

                # Place the extracted patch into the full-size patch
                patch_width = x_end_img - x_start_img
                patch_height = y_end_img - y_start_img

                if patch_width > 0 and patch_height > 0:
                    full_patch[
                        :,
                        x_offset : x_offset + patch_width,
                        y_offset : y_offset + patch_height,
                    ] = patch

                # Optionally normalize the patch by b0
                if self.normalize_to_b0:
                    full_patch = self._normalize_to_b0(full_patch)

                # Store patch and metadata
                patch_info = {
                    "patch_data": full_patch,
                    "subject_id": item["subject_id"],
                    "acquisition_id": item["acquisition_id"],
                    "slice_idx": slice_idx,
                    "patch_x": x_start,
                    "patch_y": y_start,
                    "patch_x_idx": patch_x_idx,
                    "patch_y_idx": patch_y_idx,
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

    def reconstruct_slice_from_patches(
        self, patches: List[Dict], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Reconstruct a full slice from patches using weighted averaging.

        Args:
            patches: List of patch dictionaries from create_slice_patches.
            original_shape: Original (width, height) of the slice.

        Returns:
            Reconstructed slice tensor of shape (b_values, width, height).
        """
        width, height = original_shape
        b_values = patches[0]["patch_data"].shape[0]

        # Initialize reconstruction tensors
        reconstructed = torch.zeros(
            (b_values, width, height), dtype=patches[0]["patch_data"].dtype
        )
        weights = torch.zeros(
            (b_values, width, height), dtype=patches[0]["patch_data"].dtype
        )

        stride = self.patch_size - self.patch_overlap

        for patch_info in patches:
            patch_data = patch_info["patch_data"]
            x_start = patch_info["patch_x"]
            y_start = patch_info["patch_y"]

            # Create a weight mask for this patch (higher weight in center, lower at edges)
            weight_mask = torch.ones_like(patch_data)
            if self.patch_overlap > 0:
                # Create a smooth weight mask that decreases towards the edges
                for i in range(self.patch_size):
                    for j in range(self.patch_size):
                        # Calculate distance from center
                        center_x, center_y = self.patch_size // 2, self.patch_size // 2
                        dist_x = abs(i - center_x) / (self.patch_size // 2)
                        dist_y = abs(j - center_y) / (self.patch_size // 2)
                        dist = max(dist_x, dist_y)
                        weight_mask[:, i, j] = 1.0 - 0.5 * dist

            # Add the weighted patch to the reconstruction
            x_end = min(x_start + self.patch_size, width)
            y_end = min(y_start + self.patch_size, height)

            patch_x_start = max(0, x_start)
            patch_y_start = max(0, y_start)

            x_offset = max(0, x_start - patch_x_start)
            y_offset = max(0, y_start - patch_y_start)

            patch_width = x_end - patch_x_start
            patch_height = y_end - patch_y_start

            if patch_width > 0 and patch_height > 0:
                reconstructed[:, patch_x_start:x_end, patch_y_start:y_end] += (
                    patch_data[
                        :,
                        x_offset : x_offset + patch_width,
                        y_offset : y_offset + patch_height,
                    ]
                    * weight_mask[
                        :,
                        x_offset : x_offset + patch_width,
                        y_offset : y_offset + patch_height,
                    ]
                )
                weights[:, patch_x_start:x_end, patch_y_start:y_end] += weight_mask[
                    :,
                    x_offset : x_offset + patch_width,
                    y_offset : y_offset + patch_height,
                ]

        # Normalize by weights to get the final reconstruction
        weights = torch.clamp(weights, min=1e-8)  # Prevent division by zero
        reconstructed = reconstructed / weights

        return reconstructed
