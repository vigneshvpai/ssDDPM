import os
import json
import torch
import numpy as np
from src.config.config import Config


def get_info_pt_slicewise(pt_path, data_root):
    """Get info for slicewise .pt files"""
    data = torch.load(pt_path, weights_only=False)
    image = data.get("image")
    bval = data.get("bval")

    # Extract direction and slice from filename
    filename = os.path.basename(pt_path)
    if "_x.pt" in filename:
        direction = "x"
    elif "_y.pt" in filename:
        direction = "y"
    elif "_z.pt" in filename:
        direction = "z"
    else:
        direction = "unknown"

    # Extract slice number from filename (e.g., slice00, slice01, etc.)
    slice_idx = None
    if "slice" in filename:
        try:
            slice_part = filename.split("slice")[1].split("_")[0]
            slice_idx = int(slice_part)
        except (ValueError, IndexError):
            slice_idx = None

    info = {
        "path": filename,
        "data_root": data_root,
        "image_shape": list(image.shape) if hasattr(image, "shape") else None,
        "bval": bval.tolist() if hasattr(bval, "tolist") else list(bval),
        "direction": direction,
        "slice": slice_idx,
    }
    return info


def split_dataset_slicewise(
    pt_data_root=Config.PT_DATA_ROOT_SLICEWISE,
    output_dir=os.path.join("src", "data", "dataset_split_slicewise"),
    train_ratio=0.80,
    val_ratio=0.05,
    test_ratio=0.15,
):
    """
    Split slicewise dataset into train/val/test sets.
    Creates separate JSONs for each direction (x, y, z) and an overall JSON.

    Args:
        pt_data_root: Path to slicewise PT data directory
        output_dir: Output directory for JSON files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .pt files from PT_DATA_ROOT_SLICEWISE
    pt_files = [
        os.path.join(pt_data_root, f)
        for f in os.listdir(pt_data_root)
        if f.endswith(".pt")
    ]
    pt_files.sort()  # For reproducibility

    # Separate files by direction based on filename
    x_files = [f for f in pt_files if "_x.pt" in f]
    y_files = [f for f in pt_files if "_y.pt" in f]
    z_files = [f for f in pt_files if "_z.pt" in f]

    print(f"Found {len(pt_files)} total PT files")
    print(f"Direction split: X={len(x_files)}, Y={len(y_files)}, Z={len(z_files)}")

    def split_files_by_ratio(files, train_ratio, val_ratio, test_ratio):
        """Split files into train/val/test based on ratios"""
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        return train_files, val_files, test_files

    # Split each direction
    x_train, x_val, x_test = split_files_by_ratio(
        x_files, train_ratio, val_ratio, test_ratio
    )
    y_train, y_val, y_test = split_files_by_ratio(
        y_files, train_ratio, val_ratio, test_ratio
    )
    z_train, z_val, z_test = split_files_by_ratio(
        z_files, train_ratio, val_ratio, test_ratio
    )

    # Create overall splits (all directions combined)
    all_train = x_train + y_train + z_train
    all_val = x_val + y_val + z_val
    all_test = x_test + y_test + z_test

    # Sort for reproducibility
    all_train.sort()
    all_val.sort()
    all_test.sort()

    print(
        f"Overall split: Train={len(all_train)}, Val={len(all_val)}, Test={len(all_test)}"
    )

    # Create JSON files for each direction
    directions = [
        ("x", x_train, x_val, x_test),
        ("y", y_train, y_val, y_test),
        ("z", z_train, z_val, z_test),
    ]

    for direction, train_files, val_files, test_files in directions:
        direction_output_dir = os.path.join(output_dir, direction)
        if not os.path.exists(direction_output_dir):
            os.makedirs(direction_output_dir)

        splits = [
            ("train.json", train_files),
            ("val.json", val_files),
            ("test.json", test_files),
        ]

        for fname, files in splits:
            info_list = [get_info_pt_slicewise(f, pt_data_root) for f in files]
            out_path = os.path.join(direction_output_dir, fname)
            with open(out_path, "w") as f:
                json.dump(info_list, f, indent=2)
            print(f"Saved {out_path} ({len(info_list)} samples)")

    # Create overall JSON files
    overall_splits = [
        ("train.json", all_train),
        ("val.json", all_val),
        ("test.json", all_test),
    ]

    for fname, files in overall_splits:
        info_list = [get_info_pt_slicewise(f, pt_data_root) for f in files]
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(info_list, f, indent=2)
        print(f"Saved {out_path} ({len(info_list)} samples)")

    # Print summary
    print("\n=== Dataset Split Summary ===")
    print(f"Total files processed: {len(pt_files)}")
    print(f"Output directory: {output_dir}")
    print("\nDirection-wise splits:")
    for direction, train_files, val_files, test_files in directions:
        print(
            f"  {direction.upper()}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
        )
    print(
        f"\nOverall: Train={len(all_train)}, Val={len(all_val)}, Test={len(all_test)}"
    )


if __name__ == "__main__":
    split_dataset_slicewise()
