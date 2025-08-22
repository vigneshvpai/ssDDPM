import os
import json
import torch
from src.config.config import Config


def split_dataset(
    pt_data_root=Config.PT_DATA_ROOT,
    output_dir=os.path.join("src", "data", "dataset_split"),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
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

    # Shuffle
    import random

    random.seed(seed)
    random.shuffle(pt_files)

    n_total = len(pt_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = pt_files[:n_train]
    val_files = pt_files[n_train : n_train + n_val]
    test_files = pt_files[n_train + n_val :]

    def get_info(pt_path):
        data = torch.load(pt_path, weights_only=False)
        image = data.get("image")
        bval = data.get("bval")
        info = {
            "path": pt_path,
            "file_shape": list(image.shape) if hasattr(image, "shape") else None,
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
