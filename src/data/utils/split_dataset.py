import os
import json
import torch
import nibabel as nib
import numpy as np
from src.config.config import Config


def bvals_match(bvals, ref_bvals):
    """Check if two bval lists match exactly (order and value), comparing as integers."""
    try:
        bvals_int = [int(round(float(x))) for x in bvals]
    except Exception:
        return False
    return bvals_int == ref_bvals


def width_height_match(image, required_dims):
    """Check if the width and height (first two dimensions) of the image match required_dims."""
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return tuple(image.shape[:2]) == required_dims
    return False


def split_dataset(
    pt_data_root=Config.PT_DATA_ROOT,
    original_data_root=Config.ORIGINAL_DATA_ROOT,
    output_dir=os.path.join("src", "data", "dataset_split"),
    train_ratio=0.80,
    val_ratio=0.05,
    test_ratio=0.15,
):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .pt files from PT_DATA_ROOT for train and val
    pt_files = [
        os.path.join(pt_data_root, f)
        for f in os.listdir(pt_data_root)
        if f.endswith(".pt")
    ]
    pt_files.sort()  # For reproducibility

    # List all .nii.gz files from ORIGINAL_DATA_ROOT for test
    nii_files = []
    for study_name in os.listdir(original_data_root):
        study_path = os.path.join(original_data_root, study_name)
        if not os.path.isdir(study_path):
            continue

        for f in os.listdir(study_path):
            if f.endswith(".nii.gz"):
                nii_files.append(os.path.join(study_path, f))
    nii_files.sort()  # For reproducibility

    # Filter PT files for train/val
    filtered_pt_files = []
    for pt_path in pt_files:
        try:
            data = torch.load(pt_path, weights_only=False)
            image = data.get("image")
            # Only allow if shape == EXPECTED_SHAPE
            if hasattr(image, "shape") and tuple(image.shape) == Config.EXPECTED_SHAPE:
                filtered_pt_files.append(pt_path)
        except Exception as e:
            print(f"Warning: Could not read {pt_path}: {e}")

    # Filter NIfTI files for test
    filtered_nii_files = []
    for nii_path in nii_files:
        try:
            # Load the NIfTI file to check shape
            nii_img = nib.load(nii_path)
            image_data = nii_img.get_fdata(dtype=np.float32)
            # Only allow if shape == EXPECTED_SHAPE
            if (
                hasattr(image_data, "shape")
                and tuple(image_data.shape) == Config.EXPECTED_SHAPE
            ):
                filtered_nii_files.append(nii_path)
        except Exception as e:
            print(f"Warning: Could not read {nii_path}: {e}")

    # Calculate splits for train/val from PT_DATA_ROOT
    n_total_pt = len(filtered_pt_files)
    n_train = int(n_total_pt * train_ratio)
    n_val = int(n_total_pt * val_ratio)

    train_files = filtered_pt_files[:n_train]
    val_files = filtered_pt_files[n_train : n_train + n_val]

    # Extract study_name and base_name from train/val files to avoid overlap
    used_combinations = set()
    for pt_file in train_files + val_files:
        # pt_file format: /path/to/{study_name}_{base_name}.pt
        pt_basename = os.path.basename(pt_file)
        # Remove .pt extension
        name_without_ext = pt_basename[:-3]
        # Split by the last underscore to separate study_name and base_name
        parts = name_without_ext.rsplit("_", 1)
        if len(parts) == 2:
            study_name, base_name = parts
            used_combinations.add((study_name, base_name))

    # Filter out NIfTI files that correspond to train/val files
    available_nii_files = []
    for nii_path in filtered_nii_files:
        # nii_path format: /path/to/study_name/study_name_base_name.nii.gz
        study_dir = os.path.dirname(nii_path)
        study_name = os.path.basename(study_dir)
        nii_basename = os.path.basename(nii_path)
        # Remove .nii.gz extension
        base_name = nii_basename[:-7]

        # Check if this combination is already used in train/val
        if (study_name, base_name) not in used_combinations:
            available_nii_files.append(nii_path)

    # Apply test ratio to available NIfTI files (excluding train/val overlaps)
    n_total_available = len(available_nii_files)
    n_test_nii = int(n_total_available * test_ratio)
    test_files = available_nii_files[:n_test_nii]

    def get_info_pt(pt_path, data_root):
        """Get info for .pt files"""
        data = torch.load(pt_path, weights_only=False)
        image = data.get("image")
        bval = data.get("bval")
        info = {
            "path": os.path.basename(pt_path),
            "data_root": data_root,
            "image_shape": list(image.shape) if hasattr(image, "shape") else None,
            "bval": bval.tolist() if hasattr(bval, "tolist") else list(bval),
        }
        return info

    def get_info_nii(nii_path, data_root):
        """Get info for .nii.gz files"""
        # Load the NIfTI file
        nii_img = nib.load(nii_path)
        image_data = nii_img.get_fdata(dtype=np.float32)

        # Find corresponding .bval file
        base_name = os.path.basename(nii_path)[:-7]  # remove .nii.gz
        study_dir = os.path.dirname(nii_path)
        bval_path = os.path.join(study_dir, base_name + ".bval")

        # Load bval
        bval = []
        if os.path.exists(bval_path):
            with open(bval_path, "r") as f:
                bval_line = f.readline()
                bval = [float(x) for x in bval_line.strip().split()]
        else:
            print(f"Warning: .bval file not found for {nii_path}")
            bval = [0.0] * image_data.shape[3]  # Default bvals

        # Get relative path from data_root
        rel_path = os.path.relpath(nii_path, data_root)

        info = {
            "path": rel_path,
            "data_root": data_root,
            "image_shape": (
                list(image_data.shape) if hasattr(image_data, "shape") else None
            ),
            "bval": bval,
        }
        return info

    splits = [
        ("train.json", train_files, pt_data_root, "pt"),
        ("val.json", val_files, pt_data_root, "pt"),
        ("test.json", test_files, original_data_root, "nii"),
    ]

    for fname, files, data_root, file_type in splits:
        if file_type == "pt":
            info_list = [get_info_pt(f, data_root) for f in files]
        else:  # nii
            info_list = [get_info_nii(f, data_root) for f in files]

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(info_list, f, indent=2)
        print(f"Saved {out_path} ({len(info_list)} samples) from {data_root}")


if __name__ == "__main__":
    split_dataset()
