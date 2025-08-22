import os
import torch
import numpy as np
from src.config.config import Config


def summarize_pt_data(pt_data_root=Config.PT_DATA_ROOT):
    """
    Summarizes the .pt data files in the given directory.
    Prints:
      - Number of .pt files
      - List of unique study names (prefix before first underscore)
      - Image shape distribution
      - Bval statistics (min, max, mean, std, unique counts)
    """
    if not os.path.exists(pt_data_root):
        print(f"Directory {pt_data_root} does not exist.")
        return

    pt_files = [f for f in os.listdir(pt_data_root) if f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt files found in {pt_data_root}.")
        return

    study_names = set()
    image_shapes = []
    bval_lengths = []
    bval_stats = []

    for fname in pt_files:
        # Extract study name (prefix before first underscore)
        if "_" in fname:
            study_name = fname.split("_")[0]
        else:
            study_name = fname.replace(".pt", "")
        study_names.add(study_name)

        fpath = os.path.join(pt_data_root, fname)
        try:
            data = torch.load(fpath, weights_only=False)
            img = data.get("image")
            bval = data.get("bval")
            if img is not None:
                image_shapes.append(tuple(img.shape))
            if bval is not None:
                bval = np.array(bval)
                bval_lengths.append(len(bval))
                bval_stats.append(
                    {
                        "min": float(np.min(bval)),
                        "max": float(np.max(bval)),
                        "mean": float(np.mean(bval)),
                        "std": float(np.std(bval)),
                        "unique": int(len(np.unique(bval))),
                    }
                )
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    print(f"Number of .pt files: {len(pt_files)}")
    print(f"Unique study names ({len(study_names)}): {sorted(study_names)}")

    # Image shape distribution
    from collections import Counter

    shape_counts = Counter(image_shapes)
    print("Image shape distribution:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} files")

    # Bval statistics
    if bval_stats:
        min_bval = min(stat["min"] for stat in bval_stats)
        max_bval = max(stat["max"] for stat in bval_stats)
        mean_bval = np.mean([stat["mean"] for stat in bval_stats])
        std_bval = np.mean([stat["std"] for stat in bval_stats])
        unique_bval_counts = [stat["unique"] for stat in bval_stats]
        print(f"Bval statistics across all files:")
        print(f"  Min: {min_bval}")
        print(f"  Max: {max_bval}")
        print(f"  Mean of means: {mean_bval:.2f}")
        print(f"  Mean of stds: {std_bval:.2f}")
        print(
            f"  Unique bval counts: min={min(unique_bval_counts)}, max={max(unique_bval_counts)}"
        )
    else:
        print("No bval data found.")

    # Bval length distribution
    if bval_lengths:
        length_counts = Counter(bval_lengths)
        print("Bval length distribution:")
        for length, count in length_counts.items():
            print(f"  {length}: {count} files")


if __name__ == "__main__":
    summarize_pt_data()
