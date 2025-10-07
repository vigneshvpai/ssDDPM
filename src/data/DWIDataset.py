import os
import json
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


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
            # Create tensor directly with correct dtype to avoid conversion
            image = torch.from_numpy(data_nii).float()

        # Create b_values tensor more efficiently
        b_values = torch.tensor(sample_info["bval"], dtype=torch.float32)

        # Convert to torch.Tensor if not already - use in-place conversion
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        elif image.dtype != torch.float32:
            image = image.float()  # In-place dtype conversion

        if self.preprocess_fn is not None:
            image, b_values = self.preprocess_fn(image, b_values)
        if self.transform:
            image = self.transform(image)

        # Create comprehensive metadata dictionary
        other_info = {
            "original_filename": original_filename,
            "direction": sample_info["direction"],
            "slice": sample_info["slice"],
        }

        # Add affine matrix if available
        if affine is not None:
            other_info["affine"] = affine

        return (
            image,
            b_values,
            other_info,
        )
