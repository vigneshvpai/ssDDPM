#!/usr/bin/env python3
"""
DWI Dataset Loader for DDPM Training

This module provides a PyTorch DataLoader for DWI (Diffusion Weighted Imaging) data
that has been filtered based on specific bval sequences. It loads .nii.gz files
along with their corresponding .bval and .bvec files.
"""

import os
import re
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DWIDataset(Dataset):
    """
    PyTorch Dataset for DWI data with bval/bvec filtering
    """

    def __init__(
        self,
        filtered_cases_file: str,
        transform=None,
        target_size: Tuple[int, int, int] = (64, 64, 64),
        normalize: bool = True,
        load_bvec: bool = False,
        max_cases: Optional[int] = None,
    ):
        """
        Initialize DWI Dataset

        Args:
            filtered_cases_file: Path to the filtered_cases.txt file
            transform: Optional transforms to apply to the data
            target_size: Target size for resizing (height, width, depth)
            normalize: Whether to normalize the data to [0, 1]
            load_bvec: Whether to load bvec files (for future use)
            max_cases: Maximum number of cases to load (for debugging)
        """
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.load_bvec = load_bvec

        # Parse the filtered cases file
        self.cases = self._parse_filtered_cases(filtered_cases_file)

        if max_cases is not None:
            self.cases = self.cases[:max_cases]
            logger.info(f"Limited to {max_cases} cases for debugging")

        logger.info(f"Loaded {len(self.cases)} DWI cases from {filtered_cases_file}")

    def _parse_filtered_cases(self, filtered_cases_file: str) -> List[Dict]:
        """
        Parse the filtered_cases.txt file to extract case information

        Args:
            filtered_cases_file: Path to the filtered cases file

        Returns:
            List of dictionaries containing case information
        """
        cases = []

        if not os.path.exists(filtered_cases_file):
            raise FileNotFoundError(
                f"Filtered cases file not found: {filtered_cases_file}"
            )

        with open(filtered_cases_file, "r") as f:
            content = f.read()

        # Split content into case blocks
        case_blocks = content.split("\nCase: ")[1:]  # Skip the header

        for block in case_blocks:
            lines = block.strip().split("\n")
            if not lines:
                continue

            case_name = lines[0]
            case_info = {"case_name": case_name, "files": []}

            # Extract file information
            current_files = {}
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Path:"):
                    case_info["case_path"] = line.split("Path: ")[1]
                elif line.startswith("- NII:"):
                    current_files["nii"] = line.split("- NII: ")[1]
                elif line.startswith("- BVAL:"):
                    current_files["bval"] = line.split("- BVAL: ")[1]
                elif line.startswith("- BVEC:"):
                    current_files["bvec"] = line.split("- BVEC: ")[1]
                elif line.startswith("- BVAL values:"):
                    bval_str = line.split("- BVAL values: ")[1]
                    current_files["bval_values"] = eval(bval_str)
                elif (
                    line == "" and current_files
                ):  # Empty line indicates end of file group
                    case_info["files"].append(current_files.copy())
                    current_files = {}

            # Don't forget the last file group
            if current_files:
                case_info["files"].append(current_files)

            cases.append(case_info)

        return cases

    def _load_nii_file(self, filepath: str) -> np.ndarray:
        """
        Load a .nii.gz file and return as numpy array

        Args:
            filepath: Path to the .nii.gz file

        Returns:
            Numpy array of the image data
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data
        except Exception as e:
            logger.error(f"Error loading NII file {filepath}: {e}")
            raise

    def _load_bval_file(self, filepath: str) -> np.ndarray:
        """
        Load a .bval file and return as numpy array

        Args:
            filepath: Path to the .bval file

        Returns:
            Numpy array of bval values
        """
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
            bvals = np.array([float(x) for x in content.split()])
            return bvals
        except Exception as e:
            logger.error(f"Error loading BVAL file {filepath}: {e}")
            raise

    def _load_bvec_file(self, filepath: str) -> np.ndarray:
        """
        Load a .bvec file and return as numpy array

        Args:
            filepath: Path to the .bvec file

        Returns:
            Numpy array of bvec values (3 x n_directions)
        """
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            bvec_values = []
            for line in lines:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    bvec_values.append(values)

            # Transpose to get (3, n_directions) format
            bvec_array = np.array(bvec_values).T
            return bvec_array
        except Exception as e:
            logger.error(f"Error loading BVEC file {filepath}: {e}")
            raise

    def _resize_volume(
        self, volume: np.ndarray, target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Resize a volume to target size using simple interpolation
        Handles both 3D and 4D volumes (4D volumes are common in DWI data)

        Args:
            volume: Input volume (3D or 4D)
            target_size: Target size (height, width, depth)

        Returns:
            Resized volume (3D)
        """
        from scipy.ndimage import zoom

        # Handle 4D volumes (DWI data with diffusion directions)
        if len(volume.shape) == 4:
            # For DDPM training, we'll use the b0 image (first slice) as the target
            # This is the baseline image without diffusion weighting
            volume = volume[:, :, :, 0]  # Take b0 image
            logger.info(
                f"4D DWI volume detected, using b0 image (first diffusion direction). Shape: {volume.shape}"
            )

        # Ensure we have a 3D volume
        if len(volume.shape) != 3:
            raise ValueError(
                f"Expected 3D volume after processing, got shape: {volume.shape}"
            )

        # Calculate zoom factors
        current_size = volume.shape
        zoom_factors = [
            target_size[0] / current_size[0],
            target_size[1] / current_size[1],
            target_size[2] / current_size[2],
        ]

        # Resize using scipy zoom
        resized_volume = zoom(volume, zoom_factors, order=1)  # Linear interpolation

        return resized_volume

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume to [0, 1] range

        Args:
            volume: Input volume

        Returns:
            Normalized volume
        """
        v_min = np.min(volume)
        v_max = np.max(volume)

        if v_max > v_min:
            normalized = (volume - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(volume)

        return normalized

    def __len__(self) -> int:
        """Return the number of cases in the dataset"""
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, str]]:
        """
        Get a single case from the dataset

        Args:
            idx: Index of the case to retrieve

        Returns:
            Dictionary containing the case data
        """
        case_info = self.cases[idx]
        case_name = case_info["case_name"]

        # For now, we'll use the first file in each case
        # You can modify this to use multiple files per case if needed
        file_info = case_info["files"][0]

        # Load the NII file
        nii_path = os.path.join(case_info["case_path"], file_info["nii"])
        volume = self._load_nii_file(nii_path)

        # Store original shape for reference
        original_shape = volume.shape
        logger.debug(f"Loaded volume with shape: {original_shape}")

        # Resize if needed (this will handle 4D -> 3D conversion)
        if volume.shape != self.target_size:
            volume = self._resize_volume(volume, self.target_size)

        # Normalize if requested
        if self.normalize:
            volume = self._normalize_volume(volume)

        # Convert to torch tensor
        volume_tensor = torch.from_numpy(volume).float()

        # For 2D UNet, we'll take the middle slice from the 3D volume
        if len(volume_tensor.shape) == 3:
            # Take the middle slice along the depth dimension
            middle_slice_idx = volume_tensor.shape[2] // 2
            volume_tensor = volume_tensor[
                :, :, middle_slice_idx
            ]  # Shape: (height, width)
            volume_tensor = volume_tensor.unsqueeze(
                0
            )  # Add channel dimension: (1, height, width)
            logger.debug(
                f"Extracted middle slice {middle_slice_idx} from 3D volume. New shape: {volume_tensor.shape}"
            )

        # Prepare return dictionary
        result = {
            "volume": volume_tensor,
            "case_name": case_name,
            "bval_values": np.array(file_info["bval_values"]),
            "file_path": nii_path,
            "original_shape": original_shape,  # Keep track of original shape
        }

        # Load bvec if requested
        if self.load_bvec:
            bvec_path = os.path.join(case_info["case_path"], file_info["bvec"])
            bvec_data = self._load_bvec_file(bvec_path)
            result["bvec"] = bvec_data

        # Apply transforms if any
        if self.transform:
            result = self.transform(result)

        return result


def create_dwi_dataloader(
    filtered_cases_file: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int, int] = (64, 64, 64),
    normalize: bool = True,
    max_cases: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for DWI data

    Args:
        filtered_cases_file: Path to the filtered_cases.txt file
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        target_size: Target size for resizing
        normalize: Whether to normalize the data
        max_cases: Maximum number of cases to load
        **kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader
    """
    dataset = DWIDataset(
        filtered_cases_file=filtered_cases_file,
        target_size=target_size,
        normalize=normalize,
        max_cases=max_cases,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )

    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Test the dataloader
    filtered_file = "filtered_cases.txt"

    if os.path.exists(filtered_file):
        print("Testing DWI DataLoader...")

        # Create a small test dataloader
        dataloader = create_dwi_dataloader(
            filtered_cases_file=filtered_file,
            batch_size=2,
            target_size=(32, 32, 32),
            max_cases=5,  # Only load 5 cases for testing
            num_workers=0,  # Use 0 workers for debugging
        )

        print(f"Created dataloader with {len(dataloader.dataset)} cases")

        # Test loading a batch
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Volume shape: {batch['volume'].shape}")
            print(f"  Case names: {batch['case_name']}")
            print(f"  BVAL values shape: {batch['bval_values'].shape}")

            if batch_idx >= 1:  # Only test first 2 batches
                break

        print("\nDataLoader test completed successfully!")
    else:
        print(f"Filtered cases file not found: {filtered_file}")
        print("Please run the filtering script first.")
