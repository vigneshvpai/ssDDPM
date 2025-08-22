import os
import torch
from torch.utils.data import Dataset
from src.config.config import Config


class DWI_Dataset(Dataset):
    """
    Dataset for loading .pt files containing 'image' and 'bval' keys.
    Each item is a dict with keys: 'image', 'bval', and 'filename'.
    """

    def __init__(self, pt_data_root=Config.PT_DATA_ROOT, transform=None):
        self.pt_data_root = pt_data_root
        self.transform = transform
        self.pt_files = [
            os.path.join(self.pt_data_root, f)
            for f in os.listdir(self.pt_data_root)
            if f.endswith(".pt")
        ]
        self.pt_files.sort()  # Optional: sort for reproducibility

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path, weights_only=False)
        image = data.get("image")
        bval = data.get("bval")
        sample = {"image": image, "bval": bval, "filename": os.path.basename(pt_path)}
        if self.transform:
            sample = self.transform(sample)
        return sample
