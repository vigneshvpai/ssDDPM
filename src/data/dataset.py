import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class DiffusionMRIDataset(Dataset):
    """Custom dataset for loading diffusion MRI NIfTI files, treating each 2D slice as a sample with all gradient directions as channels."""

    def __init__(self, data_dir, transform=None, target_size=(64, 64)):
        """
        Args:
            data_dir (str): Path to the data directory containing earthworm folders
            transform (callable, optional): Optional transform to be applied
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size

        # Find all NIfTI files and index all slices for all files
        self.slice_index = []  # List of (nii_file, slice_idx) tuples
        self.nii_files = []
        for earthworm_dir in self.data_dir.iterdir():
            if earthworm_dir.is_dir() and earthworm_dir.name.startswith("earthworm"):
                for file in earthworm_dir.glob("*.nii.gz"):
                    self.nii_files.append(file)
                    # Get number of slices in z for this file
                    img = nib.load(str(file))
                    data = img.get_fdata()
                    if len(data.shape) == 4:  # (x, y, z, directions)
                        num_slices = data.shape[2]
                    elif len(data.shape) == 3:  # (x, y, directions)
                        num_slices = 1
                    else:
                        raise ValueError(f"Unexpected data shape: {data.shape}")
                    for z in range(num_slices):
                        self.slice_index.append((file, z))

        print(
            f"Found {len(self.nii_files)} NIfTI files in {len(list(self.data_dir.glob('earthworm*')))} earthworm directories"
        )
        print(f"Total 2D slices indexed: {len(self.slice_index)}")

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        nii_file, slice_idx = self.slice_index[idx]

        # Load NIfTI data
        img = nib.load(str(nii_file))
        data = img.get_fdata()  # shape: (x, y, z, directions) or (x, y, directions)

        # Convert to torch tensor and normalize
        data = torch.from_numpy(data).float()

        # Normalize to [0, 1] range
        if data.max() > 0:
            data = data / data.max()

        # Extract the 2D slice with all gradient directions as channels
        # NIfTI is loaded as (width, height, slices, b-values)
        if len(data.shape) == 4:  # (width, height, slices, b-values)
            # Get all channels for this slice
            slice_2d = data[:, :, slice_idx, :]  # shape: (width, height, b-values)
            # Move channels to first dim: (b-values, width, height)
            slice_2d = slice_2d.permute(2, 0, 1)
        elif len(data.shape) == 3:  # (width, height, b-values)
            # Only one slice in z, so slice_idx should be 0
            if slice_idx != 0:
                raise IndexError("Slice index out of range for 3D data")
            slice_2d = data  # shape: (width, height, b-values)
            slice_2d = slice_2d.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Ensure slice_2d is (channels, height, width) before resizing.
        # NIfTI loads as (width, height, slices, channels), so we need (channels, height, width).
        # Use .permute(2, 1, 0) to get correct shape.
        if len(data.shape) == 4:
            slice_2d = data[:, :, slice_idx, :]  # (width, height, b-values)
            slice_2d = slice_2d.permute(2, 1, 0)  # (b-values, height, width)
        elif len(data.shape) == 3:
            if slice_idx != 0:
                raise IndexError("Slice index out of range for 3D data")
            slice_2d = data  # (width, height, b-values)
            slice_2d = slice_2d.permute(2, 1, 0)  # (b-values, height, width)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Now slice_2d is (channels, height, width)
        slice_2d = torch.nn.functional.interpolate(
            slice_2d.unsqueeze(0),  # Add batch dim
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # Remove batch dim

        # Apply transforms if any
        if self.transform:
            slice_2d = self.transform(slice_2d)

        # Return as {"images": tensor} where tensor is (channels, H, W)
        return {"images": slice_2d}
