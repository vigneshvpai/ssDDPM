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
    Visualize patches from a single .nii.gz file using improved patch creation.

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

    # Calculate patch positions using improved logic
    stride = patch_size - patch_overlap
    width, height = slice_data.shape[0], slice_data.shape[1]

    # Calculate the number of patches needed to cover the entire image
    num_patches_x = max(1, (width + stride - 1) // stride)
    num_patches_y = max(1, (height + stride - 1) // stride)

    # Prepare a colormap for different patch colors
    # We'll use a qualitative colormap for distinct colors
    from matplotlib import cm
    from matplotlib.colors import to_hex

    total_patches = num_patches_x * num_patches_y
    # Use tab20 or hsv for up to 20 or more colors
    color_map = (
        cm.get_cmap("tab20", total_patches)
        if total_patches <= 20
        else cm.get_cmap("hsv", total_patches)
    )

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
        f"Slice with Complete Patch Coverage\nPatch size: {patch_size}, Overlap: {patch_overlap}"
    )
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Draw patch boundaries using improved logic, each with a different color
    patch_count = 0
    for patch_y_idx in range(num_patches_y):
        for patch_x_idx in range(num_patches_x):
            # Calculate the starting position for this patch
            x_start = patch_x_idx * stride
            y_start = patch_y_idx * stride

            # Calculate the ending position
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            # Get color for this patch
            color = to_hex(color_map(patch_count % total_patches))

            # Create rectangle patch
            rect = patches.Rectangle(
                (x_start, y_start),
                patch_size,
                patch_size,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax2.add_patch(rect)

            # Add patch number
            ax2.text(
                x_start + patch_size // 2,
                y_start + patch_size // 2,
                str(patch_count),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight="bold",
            )
            patch_count += 1

    # Add info text
    total_patches = patch_count
    expected_patches = num_patches_x * num_patches_y
    coverage_percentage = (
        (total_patches / expected_patches) * 100 if expected_patches > 0 else 0
    )

    info_text = f"Total patches: {total_patches}\nExpected: {expected_patches}\nCoverage: {coverage_percentage:.1f}%"
    ax2.text(
        0.02,
        0.98,
        info_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Visualization saved to {save_path}")
    print(f"Total patches created: {total_patches}")
    print(f"Expected patches: {expected_patches}")
    print(f"Coverage: {coverage_percentage:.1f}%")
    print(f"Image shape: {data.shape}")
    print(f"Slice shape: {slice_data.shape}")

    return total_patches
