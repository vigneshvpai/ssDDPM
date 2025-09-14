import os
import torch
import numpy as np
import nibabel as nib
from src.config.config import Config


def convert_nii_bval_to_pt(
    input_root=Config.ORIGINAL_DATA_ROOT, output_root=Config.PT_DATA_ROOT_SLICEWISE
):
    """
    Converts all .nii.gz and .bval files under input_root to .pt files in output_root.
    For each slice, creates separate .pt files for x, y, and z directions.
    Each direction includes b0 plus the corresponding directional b-values.

    Expected image shape: (width, height, slices, b_values) = (108, 134, 25, 25)
    b-values pattern: [0, 10, 10, 10, 50, 50, 50, 80, 80, 80, ...]
    Direction mapping: b0, bx1, bx2, bx3, by1, by2, by3, bz1, bz2, bz3, ...
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

            # Expected shape: (108, 134, 25, 25)
            if img_data.shape != (108, 134, 25, 25):
                print(
                    f"Warning: Unexpected image shape {img_data.shape} for {nii_path}, expected (108, 134, 25, 25). Skipping."
                )
                continue

            # Load bval
            with open(bval_path, "r") as f:
                bval_line = f.readline()
                bval = np.array([float(x) for x in bval_line.strip().split()])

            # Verify bval has 25 values
            if (
                len(bval)
                != (Config.DWI_CONFIG["n_bvals"] * Config.DWI_CONFIG["num_dirs"]) - 2
            ):
                print(
                    f"Warning: Expected {Config.DWI_CONFIG['n_bvals']} b-values, got {len(bval)} for {nii_path}. Skipping."
                )
                continue

            # Process each slice (25 slices total)
            for slice_idx in range(Config.DWI_CONFIG["n_slices"]):
                slice_data = img_data[:, :, slice_idx, :]  # Shape: (108, 134, 25)

                # Extract b-values for each direction
                # b0 is at index 0, then x, y, z directions follow the pattern
                b0_idx = 0
                x_indices = [b0_idx] + list(
                    range(
                        1, Config.DWI_CONFIG["n_bvals"], Config.DWI_CONFIG["num_dirs"]
                    )
                )  # b0, bx1, bx2, bx3, ...
                y_indices = [b0_idx] + list(
                    range(
                        2, Config.DWI_CONFIG["n_bvals"], Config.DWI_CONFIG["num_dirs"]
                    )
                )  # b0, by1, by2, by3, ...
                z_indices = [b0_idx] + list(
                    range(
                        3, Config.DWI_CONFIG["n_bvals"], Config.DWI_CONFIG["num_dirs"]
                    )
                )  # b0, bz1, bz2, bz3, ...

                # Extract data for each direction
                x_data = slice_data[
                    :, :, x_indices
                ]  # Shape: (108, 134, 9) - b0 + 8 x-directions
                y_data = slice_data[
                    :, :, y_indices
                ]  # Shape: (108, 134, 9) - b0 + 8 y-directions
                z_data = slice_data[
                    :, :, z_indices
                ]  # Shape: (108, 134, 9) - b0 + 8 z-directions

                # Get corresponding b-values
                x_bvals = bval[x_indices]
                y_bvals = bval[y_indices]
                z_bvals = bval[z_indices]

                # Save x-direction data
                x_data_dict = {
                    "image": x_data,
                    "bval": x_bvals,
                    "direction": "x",
                    "slice": slice_idx,
                }
                x_out_fname = f"{study_name}_{base_name}_slice{slice_idx:02d}_x.pt"
                x_out_path = os.path.join(output_root, x_out_fname)
                torch.save(x_data_dict, x_out_path)

                # Save y-direction data
                y_data_dict = {
                    "image": y_data,
                    "bval": y_bvals,
                    "direction": "y",
                    "slice": slice_idx,
                }
                y_out_fname = f"{study_name}_{base_name}_slice{slice_idx:02d}_y.pt"
                y_out_path = os.path.join(output_root, y_out_fname)
                torch.save(y_data_dict, y_out_path)

                # Save z-direction data
                z_data_dict = {
                    "image": z_data,
                    "bval": z_bvals,
                    "direction": "z",
                    "slice": slice_idx,
                }
                z_out_fname = f"{study_name}_{base_name}_slice{slice_idx:02d}_z.pt"
                z_out_path = os.path.join(output_root, z_out_fname)
                torch.save(z_data_dict, z_out_path)

                print(
                    f"Saved slice {slice_idx}: {x_out_fname}, {y_out_fname}, {z_out_fname}"
                )

            print(
                f"Completed processing {nii_path}: 25 slices × 3 directions = 75 .pt files"
            )


if __name__ == "__main__":
    convert_nii_bval_to_pt()
