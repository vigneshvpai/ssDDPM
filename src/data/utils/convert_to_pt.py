import os
import torch
import numpy as np
import nibabel as nib
from src.config.config import Config


def convert_nii_bval_to_pt(
    input_root=Config.ORIGINAL_DATA_ROOT, output_root=Config.PT_DATA_ROOT
):
    """
    Converts all .nii.gz and .bval files under input_root to .pt files in output_root.
    Each .pt file contains a dict with keys: 'image' (numpy array), 'bval' (numpy array).
    Ignores .bvec files.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for study_name in os.listdir(input_root):
        study_path = os.path.join(input_root, study_name)
        if not os.path.isdir(study_path):
            continue

        nii_files = [f for f in os.listdir(study_path) if f.endswith(".nii.gz")]
        for nii_file in nii_files:
            base_name = nii_file[:-7]  # remove .nii.gz
            nii_path = os.path.join(study_path, nii_file)
            bval_path = os.path.join(study_path, base_name + ".bval")
            # Check if .bval exists
            if not os.path.exists(bval_path):
                print(f"Warning: .bval file not found for {nii_path}, skipping.")
                continue

            # Load image
            img = nib.load(nii_path)
            img_data = img.get_fdata(dtype=np.float32)

            # Load bval
            with open(bval_path, "r") as f:
                bval_line = f.readline()
                bval = np.array([float(x) for x in bval_line.strip().split()])

            # Prepare dict
            data = {"image": img_data, "bval": bval}

            # Save as .pt
            out_fname = f"{study_name}_{base_name}.pt"
            out_path = os.path.join(output_root, out_fname)
            torch.save(data, out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    convert_nii_bval_to_pt()
