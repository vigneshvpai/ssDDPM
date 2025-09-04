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

        if self.preprocess_fn is not None:
            image, min_val, max_val = self.preprocess_fn(image)
        if self.transform:
            image = self.transform(image)

        if affine is not None:
            other_info = {"affine": affine, "min_val": min_val, "max_val": max_val}
        else:
            other_info = {"min_val": min_val, "max_val": max_val}

        return image, b_values, other_info
