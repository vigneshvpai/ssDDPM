import os
import json
import torch
from torch.utils.data import Dataset


class DWIDataset(Dataset):
    """
    PyTorch Dataset for loading DWI data using split JSONs.
    Each JSON entry should have: 'path', 'image_shape', 'bval'.
    """

    def __init__(
        self,
        split_json_path=None,
        transform=None,
        preprocess_fn=None,
    ):
        self.split_json_path = split_json_path
        self.transform = transform
        self.preprocess_fn = preprocess_fn

        # Load the list of samples from the split JSON
        with open(self.split_json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        pt_path = sample_info["path"]
        data = torch.load(pt_path, weights_only=False)

        image = data.get("image")
        b_values = torch.tensor(sample_info["bval"])
        b_values = b_values.repeat(image.shape[3])

        # Convert to torch.Tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
        if self.transform:
            image = self.transform(image)

        return image, b_values
