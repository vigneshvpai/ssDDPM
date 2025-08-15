import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings


class MedicalDataLoader:
    """
    Utility class for loading and processing medical imaging data.
    
    This class handles loading of DWI (Diffusion Weighted Imaging) data with
    different NEX (Number of EXcitations) values and provides utilities for
    creating NEX=6 images from NEX=1 acquisitions.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the medical data loader.
        
        Args:
            data_dir: Directory containing medical imaging data
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.supported_formats = ['.nii', '.nii.gz', '.hdr', '.img']
    
    def load_dwi_data(self, file_path: str, normalize: bool = True) -> torch.Tensor:
        """
        Load DWI data from file.
        
        Args:
            file_path: Path to the DWI data file
            normalize: Whether to normalize the data to [0, 1] range
            
        Returns:
            Tensor containing the DWI data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Load data using nibabel
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Error loading data from {file_path}: {e}")
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        # Normalize if requested
        if normalize:
            data = self._normalize_data(data)
        
        return data
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Input data tensor
            
        Returns:
            Normalized data tensor
        """
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def create_nex6_from_nex1_repetitions(self, nex1_files: List[str], 
                                        output_path: Optional[str] = None) -> torch.Tensor:
        """
        Create NEX=6 images by averaging multiple NEX=1 acquisitions.
        
        Args:
            nex1_files: List of file paths for NEX=1 acquisitions
            output_path: Path to save the averaged data (optional)
            
        Returns:
            Tensor containing the averaged NEX=6 data
        """
        if len(nex1_files) < 6:
            warnings.warn(f"Only {len(nex1_files)} NEX=1 files provided, expected 6")
        
        # Load all NEX=1 acquisitions
        nex1_data = []
        for file_path in nex1_files:
            data = self.load_dwi_data(file_path, normalize=True)
            nex1_data.append(data)
        
        # Stack and average
        nex1_stack = torch.stack(nex1_data, dim=0)
        nex6_data = torch.mean(nex1_stack, dim=0, keepdim=True)
        
        # Save if output path provided
        if output_path:
            self._save_data(nex6_data, output_path)
        
        return nex6_data
    
    def _save_data(self, data: torch.Tensor, output_path: str):
        """
        Save data to file.
        
        Args:
            data: Data tensor to save
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy and save
        data_np = data.numpy()
        
        if output_path.suffix in ['.nii', '.nii.gz']:
            # Create NIfTI image
            img = nib.Nifti1Image(data_np, np.eye(4))
            nib.save(img, str(output_path))
        else:
            # Save as numpy array
            np.save(str(output_path), data_np)
    
    def extract_slices(self, data: torch.Tensor, slice_indices: Optional[List[int]] = None,
                      axis: int = 2) -> torch.Tensor:
        """
        Extract 2D slices from 3D volume data.
        
        Args:
            data: 3D volume data
            slice_indices: Indices of slices to extract (if None, extract all)
            axis: Axis along which to extract slices
            
        Returns:
            Tensor containing 2D slices
        """
        if data.dim() != 3:
            raise ValueError(f"Expected 3D data, got {data.dim()}D")
        
        if slice_indices is None:
            # Extract all slices
            if axis == 0:
                slices = [data[i, :, :] for i in range(data.shape[0])]
            elif axis == 1:
                slices = [data[:, i, :] for i in range(data.shape[1])]
            elif axis == 2:
                slices = [data[:, :, i] for i in range(data.shape[2])]
            else:
                raise ValueError(f"Invalid axis: {axis}")
        else:
            # Extract specific slices
            if axis == 0:
                slices = [data[i, :, :] for i in slice_indices if i < data.shape[0]]
            elif axis == 1:
                slices = [data[:, i, :] for i in slice_indices if i < data.shape[1]]
            elif axis == 2:
                slices = [data[:, :, i] for i in slice_indices if i < data.shape[2]]
            else:
                raise ValueError(f"Invalid axis: {axis}")
        
        return torch.stack(slices, dim=0)
    
    def preprocess_for_training(self, nex1_data: torch.Tensor, nex6_data: torch.Tensor,
                              target_size: Tuple[int, int] = (64, 64),
                              augment: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess data for training the diffusion model.
        
        Args:
            nex1_data: NEX=1 data tensor
            nex6_data: NEX=6 data tensor
            target_size: Target size for resizing
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (processed_nex1, processed_nex6)
        """
        # Ensure data has correct shape
        if nex1_data.dim() == 3:
            nex1_data = nex1_data.unsqueeze(0)  # Add batch dimension
        if nex6_data.dim() == 3:
            nex6_data = nex6_data.unsqueeze(0)  # Add batch dimension
        
        # Resize to target size
        nex1_resized = self._resize_data(nex1_data, target_size)
        nex6_resized = self._resize_data(nex6_data, target_size)
        
        # Apply augmentation if requested
        if augment:
            nex1_resized = self._apply_augmentation(nex1_resized)
            nex6_resized = self._apply_augmentation(nex6_resized)
        
        return nex1_resized, nex6_resized
    
    def _resize_data(self, data: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize data to target size using bilinear interpolation.
        
        Args:
            data: Input data tensor
            target_size: Target size (height, width)
            
        Returns:
            Resized data tensor
        """
        if data.dim() == 4:  # Batch, channels, height, width
            resized = torch.nn.functional.interpolate(
                data, size=target_size, mode='bilinear', align_corners=False
            )
        else:
            # Add batch and channel dimensions if needed
            if data.dim() == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            elif data.dim() == 3:
                data = data.unsqueeze(0)
            
            resized = torch.nn.functional.interpolate(
                data, size=target_size, mode='bilinear', align_corners=False
            )
        
        return resized
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation.
        
        Args:
            data: Input data tensor
            
        Returns:
            Augmented data tensor
        """
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            data = torch.flip(data, dims=[-1])
        
        # Random vertical flip
        if torch.rand(1) < 0.5:
            data = torch.flip(data, dims=[-2])
        
        # Random rotation (90 degree increments)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            data = torch.rot90(data, k, dims=[-2, -1])
        
        return data
    
    def calculate_snr(self, clean_data: torch.Tensor, noisy_data: torch.Tensor) -> float:
        """
        Calculate Signal-to-Noise Ratio between clean and noisy data.
        
        Args:
            clean_data: Clean reference data
            noisy_data: Noisy data
            
        Returns:
            SNR value in dB
        """
        signal_power = torch.mean(clean_data ** 2)
        noise_power = torch.mean((noisy_data - clean_data) ** 2)
        
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr.item()
    
    def visualize_data(self, nex1_data: torch.Tensor, nex6_data: torch.Tensor,
                      slice_idx: int = 0, save_path: Optional[str] = None):
        """
        Visualize NEX=1 and NEX=6 data for comparison.
        
        Args:
            nex1_data: NEX=1 data tensor
            nex6_data: NEX=6 data tensor
            slice_idx: Index of slice to visualize
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract slice
        if nex1_data.dim() == 4:
            nex1_slice = nex1_data[0, 0, :, :] if nex1_data.shape[1] == 1 else nex1_data[0, slice_idx, :, :]
        else:
            nex1_slice = nex1_data[slice_idx, :, :]
        
        if nex6_data.dim() == 4:
            nex6_slice = nex6_data[0, 0, :, :] if nex6_data.shape[1] == 1 else nex6_data[0, slice_idx, :, :]
        else:
            nex6_slice = nex6_data[slice_idx, :, :]
        
        # Calculate difference
        diff_slice = nex6_slice - nex1_slice
        
        # Plot
        im1 = axes[0].imshow(nex1_slice, cmap='gray')
        axes[0].set_title('NEX=1 (Low SNR)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(nex6_slice, cmap='gray')
        axes[1].set_title('NEX=6 (High SNR)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(diff_slice, cmap='RdBu_r')
        axes[2].set_title('Difference (NEX=6 - NEX=1)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_clinical_dataset(self, dataset_path: str) -> Dict[str, torch.Tensor]:
        """
        Load a clinical dataset with NEX=1 and NEX=6 data.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary containing loaded data
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Look for NEX=1 and NEX=6 data
        nex1_files = list(dataset_path.glob("*nex1*"))
        nex6_files = list(dataset_path.glob("*nex6*"))
        
        if not nex1_files:
            raise FileNotFoundError(f"No NEX=1 files found in {dataset_path}")
        
        # Load NEX=1 data
        nex1_data = []
        for file_path in nex1_files:
            data = self.load_dwi_data(str(file_path))
            nex1_data.append(data)
        
        # Load or create NEX=6 data
        if nex6_files:
            nex6_data = []
            for file_path in nex6_files:
                data = self.load_dwi_data(str(file_path))
                nex6_data.append(data)
        else:
            # Create NEX=6 by averaging NEX=1
            print("No NEX=6 files found, creating by averaging NEX=1 data...")
            nex6_data = [self.create_nex6_from_nex1_repetitions([str(f) for f in nex1_files])]
        
        return {
            "nex1_data": torch.stack(nex1_data, dim=0),
            "nex6_data": torch.stack(nex6_data, dim=0),
            "nex1_files": nex1_files,
            "nex6_files": nex6_files if nex6_files else None
        }


def create_synthetic_clinical_data(num_subjects: int = 10, 
                                 slices_per_subject: int = 20,
                                 image_size: Tuple[int, int] = (128, 128)) -> Dict[str, torch.Tensor]:
    """
    Create synthetic clinical data for testing and development.
    
    Args:
        num_subjects: Number of subjects
        slices_per_subject: Number of slices per subject
        image_size: Size of each slice
        
    Returns:
        Dictionary containing synthetic data
    """
    total_slices = num_subjects * slices_per_subject
    
    # Create synthetic brain-like structures
    nex1_data = torch.zeros(total_slices, 1, image_size[0], image_size[1])
    nex6_data = torch.zeros(total_slices, 1, image_size[0], image_size[1])
    
    for i in range(total_slices):
        # Create base brain structure
        base_image = torch.randn(image_size[0], image_size[1]) * 0.1
        
        # Add tissue structures
        # White matter (brighter regions)
        white_matter_mask = torch.zeros(image_size[0], image_size[1])
        white_matter_mask[image_size[0]//4:3*image_size[0]//4, 
                         image_size[1]//4:3*image_size[1]//4] = 1
        white_matter = torch.randn(image_size[0], image_size[1]) * 0.3 * white_matter_mask
        
        # Gray matter (medium intensity)
        gray_matter = torch.randn(image_size[0], image_size[1]) * 0.2 * (1 - white_matter_mask)
        
        # CSF (dark regions)
        csf_mask = torch.zeros(image_size[0], image_size[1])
        csf_mask[image_size[0]//8:image_size[0]//8+10, 
                image_size[1]//8:image_size[1]//8+10] = 1
        csf = torch.randn(image_size[0], image_size[1]) * 0.05 * csf_mask
        
        # Combine structures
        clean_image = base_image + white_matter + gray_matter + csf
        
        # Add some lesion-like structures randomly
        if torch.rand(1) < 0.2:
            lesion_mask = torch.zeros(image_size[0], image_size[1])
            lesion_x = torch.randint(image_size[0]//4, 3*image_size[0]//4, (1,))
            lesion_y = torch.randint(image_size[1]//4, 3*image_size[1]//4, (1,))
            lesion_mask[lesion_y-3:lesion_y+3, lesion_x-3:lesion_x+3] = 1
            clean_image += lesion_mask * torch.randn(image_size[0], image_size[1]) * 0.4
        
        # Create NEX=1 (noisy) and NEX=6 (clean) versions
        nex1_noise = torch.randn(image_size[0], image_size[1]) * 0.2
        nex6_noise = torch.randn(image_size[0], image_size[1]) * 0.05  # Less noise for NEX=6
        
        nex1_data[i, 0] = clean_image + nex1_noise
        nex6_data[i, 0] = clean_image + nex6_noise
    
    return {
        "nex1_data": nex1_data,
        "nex6_data": nex6_data,
        "num_subjects": num_subjects,
        "slices_per_subject": slices_per_subject,
        "image_size": image_size
    }
