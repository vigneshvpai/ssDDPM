import os
import json
import torch
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
    output_dir=os.path.join("src", "data", "dataset_split"),
    train_ratio=0.80,
    val_ratio=0.05,
    test_ratio=0.15,
):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .pt files
    pt_files = [
        os.path.join(pt_data_root, f)
        for f in os.listdir(pt_data_root)
        if f.endswith(".pt")
    ]
    pt_files.sort()  # For reproducibility

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

    n_total = len(filtered_pt_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = filtered_pt_files[:n_train]
    val_files = filtered_pt_files[n_train : n_train + n_val]
    test_files = filtered_pt_files[n_train + n_val :]

    def get_info(pt_path):
        data = torch.load(pt_path, weights_only=False)
        image = data.get("image")
        bval = data.get("bval")
        info = {
            "path": os.path.basename(pt_path),
            "image_shape": list(image.shape) if hasattr(image, "shape") else None,
            "bval": bval.tolist() if hasattr(bval, "tolist") else list(bval),
        }
        return info

    splits = [
        ("train.json", train_files),
        ("val.json", val_files),
        ("test.json", test_files),
    ]

    for fname, files in splits:
        info_list = [get_info(f) for f in files]
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(info_list, f, indent=2)
        print(f"Saved {out_path} ({len(info_list)} samples)")


if __name__ == "__main__":
    split_dataset()
