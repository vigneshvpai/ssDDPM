#!/usr/bin/env python3

import json
from src.config import Config
from src.data.dataset import DWIDataset
from src.visualization.patch_visualizer import visualize_nii_patches


def main():
    # Load your data list (replace with your actual data list path)
    data_list_path = "src/data/metadata/train_data.json"

    # Create dataset instance
    dataset = DWIDataset(
        data_list_path=data_list_path,
        patch_size=Config.PATCH_SIZE,
        patch_overlap=Config.PATCH_OVERLAP,
        max_b_values=25,
    )

    # Get the first item from your data list
    first_item = dataset.data_list[0]
    print(f"Visualizing: {first_item['nii_path']}")
    print(f"Subject ID: {first_item['subject_id']}")
    print(f"Acquisition ID: {first_item['acquisition_id']}")

    # Visualize the patches for this .nii.gz file
    visualize_nii_patches(
        nii_path=first_item["nii_path"],
        slice_idx=0,  # First slice
        b_value_idx=0,  # First b-value (b0)
        patch_size=Config.PATCH_SIZE,
        patch_overlap=Config.PATCH_OVERLAP,
    )

    # Print dataset info
    info = dataset.get_data_info()
    print(f"\nDataset Info:")
    print(f"Total patches: {info['num_patches']}")
    print(f"Number of subjects: {info['num_subjects']}")
    print(f"Number of acquisitions: {info['num_acquisitions']}")


if __name__ == "__main__":
    main()
