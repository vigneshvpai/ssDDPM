import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class DiffusionMRIDataset(Dataset):
    """Custom dataset for loading diffusion MRI NIfTI files"""

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

        # Find all NIfTI files
        self.nii_files = []
        for earthworm_dir in self.data_dir.iterdir():
            if earthworm_dir.is_dir() and earthworm_dir.name.startswith("earthworm"):
                for file in earthworm_dir.glob("*.nii.gz"):
                    self.nii_files.append(file)

        print(
            f"Found {len(self.nii_files)} NIfTI files in {len(list(self.data_dir.glob('earthworm*')))} earthworm directories"
        )

    def __len__(self):
        return len(self.nii_files)

    def __getitem__(self, idx):
        nii_file = self.nii_files[idx]

        # Load NIfTI data
        img = nib.load(str(nii_file))
        data = img.get_fdata()

        # Convert to torch tensor and normalize
        data = torch.from_numpy(data).float()

        # Normalize to [0, 1] range
        if data.max() > 0:
            data = data / data.max()

        # Take a middle slice for 2D processing (you can modify this for 3D)
        # For now, we'll use the middle slice of the first diffusion direction
        if len(data.shape) == 4:  # (x, y, z, diffusion_directions)
            middle_z = data.shape[2] // 2
            data = data[
                :, :, middle_z, 0
            ]  # Take middle z-slice, first diffusion direction
        elif len(data.shape) == 3:  # (x, y, diffusion_directions)
            data = data[:, :, 0]  # Take first diffusion direction

        # Ensure 2D
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")

        # Resize to target size
        data = torch.nn.functional.interpolate(
            data.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # Remove batch and channel dims

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return {"images": data.unsqueeze(0)}  # Add channel dimension
