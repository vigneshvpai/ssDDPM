#!/usr/bin/env python3

import json
import os
from src.config import Config
from src.data.dataloader import DWIDataset
from src.visualization.patch_visualizer import visualize_nii_patches


def main():
    # Load your data list (replace with your actual data list path)
    data_list_path = Config.TRAIN_DATA_LIST

    # Check if the data list file exists
    if not os.path.exists(data_list_path):
        print(f"Error: Data list file not found at {data_list_path}")
        return

    try:
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

        # Check if the NIfTI file exists
        if not os.path.exists(first_item["nii_path"]):
            print(f"Error: NIfTI file not found at {first_item['nii_path']}")
            print("Please check the file path in your data list.")
            return

        # Create output directory for plots
        os.makedirs("tests/test_plots", exist_ok=True)

        # Visualize the patches for this .nii.gz file
        total_patches = visualize_nii_patches(
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

        # Get coverage analysis
        coverage_stats = dataset.analyze_patch_coverage()
        print(f"\nCoverage Analysis:")
        print(f"Analyzed {len(coverage_stats)} slice-acquisition combinations")

        # Show some coverage statistics
        coverage_percentages = [
            stats["coverage_percentage"] for stats in coverage_stats.values()
        ]
        if coverage_percentages:
            print(
                f"Average coverage: {sum(coverage_percentages) / len(coverage_percentages):.1f}%"
            )
            print(f"Min coverage: {min(coverage_percentages):.1f}%")
            print(f"Max coverage: {max(coverage_percentages):.1f}%")

        # Show details for the first few items
        print(f"\nSample coverage details:")
        for i, (key, stats) in enumerate(list(coverage_stats.items())[:3]):
            print(
                f"  {key}: {stats['width']}x{stats['height']} -> {stats['actual_total']} patches ({stats['coverage_percentage']:.1f}%)"
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
