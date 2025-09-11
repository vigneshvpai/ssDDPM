import os
import json
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from src.config.config import Config


class DWIDataset(Dataset):
    """
    PyTorch Dataset for loading DWI data using split JSONs.
    Each JSON entry should have: 'path', 'image_shape', 'bval'.
    """

    def __init__(
        self,
        split_json_path=None,
        data_root=None,
        transform=None,
        preprocess_fn=None,
    ):
        self.split_json_path = split_json_path
        self.data_root = data_root
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.num_dirs = Config.ADC_CONFIG["num_dirs"]
        # Load the list of samples from the split JSON
        with open(self.split_json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        path = os.path.join(self.data_root, sample_info["path"])

        # Extract original filename for saving
        original_filename = os.path.basename(sample_info["path"])
        # Remove extension if it exists
        if original_filename.endswith(".nii.gz"):
            original_filename = original_filename[:-7]  # Remove .nii.gz
        elif original_filename.endswith(".pt"):
            original_filename = original_filename[:-3]  # Remove .pt

        affine = None
        if path.endswith(".pt"):
            data = torch.load(path, weights_only=False)
            image = data.get("image")
        elif path.endswith(".nii.gz"):
            # Load the NIfTI file
            nii_img = nib.load(path)
            data_nii = nii_img.get_fdata(dtype=np.float32)
            # Get the affine matrix
            affine = nii_img.affine
            image = torch.from_numpy(data_nii)

        b_values = torch.tensor(sample_info["bval"])
        b_values = b_values.repeat(image.shape[3])

        # Convert to torch.Tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        # Split by directions: b0 + (bx, by, bz) groups
        # More efficient vectorized approach

        b0_idx = 0  # First volume is always b0
        num_directions = self.num_dirs  # x, y, z
        total_volumes = image.shape[3]

        # Use sample index to deterministically select direction
        # This ensures all directions are used evenly across the dataset
        direction = idx % self.num_dirs

        # Create indices for the selected direction
        direction_indices = torch.cat(
            [
                torch.tensor([b0_idx]),  # b0 volume
                torch.arange(
                    1 + direction, total_volumes, num_directions
                ),  # direction volumes
            ]
        )
        # Extract volumes and b-values for this direction
        direction_volumes = image[:, :, :, direction_indices]
        direction_bvals = b_values[direction_indices]

        # Apply preprocessing if needed
        if self.preprocess_fn is not None:
            direction_volumes, min_val, max_val = self.preprocess_fn(direction_volumes)
        if self.transform:
            direction_volumes = self.transform(direction_volumes)

        # Prepare metadata
        if affine is not None:
            other_info = {
                "affine": affine,
                "min_val": min_val,
                "max_val": max_val,
                "original_filename": original_filename,
                "direction": direction,  # Add direction info
            }
        else:
            other_info = {
                "min_val": min_val,
                "max_val": max_val,
                "original_filename": original_filename,
                "direction": direction,  # Add direction info
            }

        return direction_volumes, direction_bvals, other_info
