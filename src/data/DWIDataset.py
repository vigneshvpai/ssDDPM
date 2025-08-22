import os
import json
import torch
from torch.utils.data import Dataset
from src.config.config import Config
from src.data.Preprocess import Preprocess


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

        # Use Preprocess().preprocess as default if preprocess_fn is None
        if preprocess_fn is None:
            self.preprocess_fn = Preprocess().preprocess
        else:
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
        bval = data.get("bval")
        # Convert to torch.Tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(bval, torch.Tensor):
            bval = torch.from_numpy(bval)
        sample = {
            "image": image,
            "bval": bval,
            "filename": os.path.basename(pt_path),
            "image_shape": sample_info.get("image_shape"),
        }
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(sample)
        if self.transform:
            image = self.transform(image)
        return image
