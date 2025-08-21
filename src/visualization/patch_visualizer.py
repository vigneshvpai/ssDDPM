"""
Visualization utilities for DWI patch extraction.
"""

import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from typing import Optional


def visualize_nii_patches(
    nii_path: str,
    slice_idx: int = 0,
    b_value_idx: int = 0,
    patch_size: int = 128,
    patch_overlap: int = 32,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 8),
):
    """
    Visualize patches from a single .nii.gz file.

    Args:
        nii_path: Path to .nii.gz file
        slice_idx: Which slice to visualize (default: 0)
        b_value_idx: Which b-value to visualize (default: 0)
        patch_size: Size of patches
        patch_overlap: Overlap between patches
        save_path: Path to save the visualization (default: tests/test_plots/nii_patches_visualization.png)
        figsize: Figure size for the plot
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = os.path.join("tests", "test_plots", "nii_patches_visualization.png")

    # Load the .nii.gz file
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Convert to torch tensor
    data = torch.from_numpy(data).float()

    # Ensure correct shape: (width, height, slices, b_values) -> (b_values, width, height, slices)
    if len(data.shape) == 4:
        data = data.permute(3, 0, 1, 2)

    # Extract the specific slice and b-value
    slice_data = data[b_value_idx, :, :, slice_idx].numpy()

    # Calculate patch positions
    stride = patch_size - patch_overlap
    width, height = slice_data.shape[0], slice_data.shape[1]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Original slice
    im1 = ax1.imshow(slice_data, cmap="gray")
    ax1.set_title(
        f"Original Slice\nShape: {slice_data.shape}\nB-value index: {b_value_idx}, Slice: {slice_idx}"
    )
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Plot 2: Slice with patch grid
    im2 = ax2.imshow(slice_data, cmap="gray")
    ax2.set_title(
        f"Slice with Patch Grid\nPatch size: {patch_size}, Overlap: {patch_overlap}"
    )
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Draw patch boundaries
    patch_count = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Create rectangle patch
            rect = patches.Rectangle(
                (x, y),
                patch_size,
                patch_size,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            ax2.add_patch(rect)

            # Add patch number
            ax2.text(
                x + patch_size // 2,
                y + patch_size // 2,
                str(patch_count),
                ha="center",
                va="center",
                color="red",
                fontsize=8,
                fontweight="bold",
            )
            patch_count += 1

    # Add info text
    total_patches = patch_count
    ax2.text(
        0.02,
        0.98,
        f"Total patches: {total_patches}",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Visualization saved to {save_path}")
    print(f"Total patches created: {total_patches}")
    print(f"Image shape: {data.shape}")
    print(f"Slice shape: {slice_data.shape}")

    return total_patches
