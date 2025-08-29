import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.trainer import Trainer
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
        map_location="cuda:0",
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
    data_module = DWIDataLoader(test_json=Config.TEST_NIFTI_JSON_PATH)

    print("Running inference on test set...")

    # Get a batch from test set
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()

    # Get first batch
    batch = next(iter(test_dataloader))
    original_images, b_values, other_info = batch

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    original_images = original_images.to(device)
    b_values = b_values.to(device)

    print(f"Test batch shape: {original_images.shape}")
    print(f"B-values shape: {b_values.shape}")

    # Run inference
    with torch.no_grad():
        generated_images = model.inference(original_images, b_values)

    print(f"Generated shape: {generated_images.shape}")

    for i in range(generated_images.shape[0]):
        generated_image = generated_images[i]

        min_val = other_info["min_val"][i]
        max_val = other_info["max_val"][i]

        generated_image = Postprocess.denormalize_from_b0(
            Postprocess.unpad_from_unet_compatible(
                Postprocess.unflatten_slices_and_bvals(generated_image)
            ),
            min_val,
            max_val,
        )

        print(f"Generated shape after postprocessing: {generated_image.shape}")
        # Save results
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)

        # Save generated images
        generated_filename = save_dir / f"generated_image_{i}.nii.gz"
        affine = other_info["affine"][i]
        save_as_nifti(generated_image, generated_filename, affine)

        print(f"Results saved to: {generated_filename}")

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
