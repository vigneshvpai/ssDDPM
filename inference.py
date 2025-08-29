import lightning as L
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from src.model.SSDDPM import SSDDPM
from src.data.DWIDataLoader import DWIDataLoader
from src.config.config import Config
from src.data.Postprocess import Postprocess


def load_model_from_checkpoint(checkpoint_path):
    """Load the trained model from checkpoint using Lightning."""
    model = SSDDPM.load_from_checkpoint(
        checkpoint_path,
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
    )
    return model


def save_as_nifti(data, filename, affine=None):
    """Save data as .nii.gz file."""
    # Convert to numpy and ensure correct data type
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Create default affine if not provided
    if affine is None:
        affine = np.eye(4)

    # Create NIfTI image
    nii_img = nib.Nifti1Image(data, affine)

    # Save as .nii.gz
    nib.save(nii_img, filename)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run SSDDPM inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="inference_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model using Lightning
    model = load_model_from_checkpoint(args.checkpoint)

    # Create data module
    data_module = DWIDataLoader(test_json_path=Config.TEST_NIFTI_JSON_PATH)

    print("Running inference on test set...")

    # Get a batch from test set
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()

    # Get first batch
    batch = next(iter(test_dataloader))
    original_images, b_values, other_info = batch

    print(f"Test batch shape: {original_images.shape}")
    print(f"B-values shape: {b_values.shape}")

    # Run inference
    with torch.no_grad():
        generated_images = model.inference(original_images, b_values)

    print(f"Generated shape: {generated_images.shape}")

    generated_images = Postprocess.denormalize_from_b0(
        Postprocess.unpad_from_unet_compatible(
            Postprocess.unflatten_slices_and_bvals(generated_images)
        ),
        other_info["min_val"],
        other_info["max_val"],
    )

    print(f"Generated shape after postprocessing: {generated_images.shape}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save generated images
    generated_filename = save_dir / "generated_images.nii.gz"
    save_as_nifti(generated_images, generated_filename, other_info["affine"])

    print(f"Results saved to: {generated_filename}")

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
