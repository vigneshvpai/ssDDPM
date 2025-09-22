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

    # Use the first available GPU or CPU if no GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SSDDPM.load_from_checkpoint(
        checkpoint_path,
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
        map_location=device,
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
    data_module = DWIDataLoader(
        test_json=Config.TEST_SPLIT_JSON, data_root=Config.ORIGINAL_DATA_ROOT
    )

    print("Running inference on test set...")

    # Get all batches from test set
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()

    # Move tensors to the same device as the model
    device = next(model.parameters()).device

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    total_batches = len(test_dataloader)
    print(f"Processing {total_batches} batches...")

    # Process all batches
    for batch_idx, batch in enumerate(test_dataloader):
        print(f"Processing batch {batch_idx + 1}/{total_batches}")

        original_images, b_values, other_info = batch

        # Move tensors to the same device as the model
        original_images = original_images.to(device)
        b_values = b_values.to(device)

        # Run inference
        with torch.no_grad():
            generated_images = model.inference(original_images, b_values)

        # Process each image in the batch
        for i in range(generated_images.shape[0]):
            generated_image = generated_images[i]

            min_val = other_info["min_val"][i]
            max_val = other_info["max_val"][i]
            original_filename = other_info["original_filename"][i]

            generated_image = Postprocess.denormalize_from_b0(
                Postprocess.unpad_from_unet_compatible(
                    Postprocess.unflatten_slices_and_bvals(generated_image)
                ),
                min_val,
                max_val,
            )

            # Transpose dimensions so that slices and bvalues are swapped for NIfTI viewing
            # From (width, height, slices, bvalues) to (width, height, bvalues, slices)
            generated_image = generated_image.permute(0, 1, 3, 2)

            # Save generated images with original filename + __DENOISED
            generated_filename = save_dir / f"{original_filename}__DENOISED.nii.gz"
            affine = other_info["affine"][i]
            save_as_nifti(generated_image, generated_filename, affine)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
