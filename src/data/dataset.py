"""
PyTorch Dataset for 2D DWI data processing.
Converts 3D DWI volumes into 2D slices and generates patches for 2D UNet training.

This module maintains backward compatibility by importing from the new modular structure.
"""

# Import from the new modular structure
from src.data.dataloader import DWIDataset, DWIDataLoader

# Re-export for backward compatibility
__all__ = ["DWIDataset", "DWIDataLoader"]
