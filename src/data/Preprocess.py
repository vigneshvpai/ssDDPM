import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self):
        pass

    def normalize_to_b0(self, image):
        """
        Normalize the image to the 0-1 range globally.
        Args:
            image (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The image scaled to 0-1.
        """
        min_val = image.min()
        max_val = image.max()
        scale = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
        image_norm = (image - min_val) / scale
        return image_norm, min_val, max_val

    def reorder_slices_and_bvals(self, image):
        # Expecting image shape: (width, height, slices, bvalues)
        if image.ndim != 4:
            raise ValueError(
                f"Expected image of shape (width, height, slices, bvalues), got {image.shape}"
            )
        # Permute to (slices, bvalues, height, width)
        image = image.permute(2, 3, 1, 0)

        return image

    def pad_to_unet_compatible(self, image, target_shape=None):
        """
        Pad the image tensor with zeros to make width and height match target_shape.
        Args:
            image (torch.Tensor): Image tensor of shape (width, height, slices, bvals).
            target_shape (tuple): (target_height, target_width)
        Returns:
            torch.Tensor: Zero-padded image tensor.
        """
        if target_shape is None:
            target_shape = Config.UNET_COMPATIBLE_SHAPE

        # image shape: (width, height, slices, bvals)
        w, h, s, b = image.shape
        target_h, target_w = target_shape

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # Calculate padding for width and height dimensions
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h

        # For padding width and height (first two dimensions)
        pad = (0, 0, 0, 0, pad_left_h, pad_right_h, pad_left_w, pad_right_w)
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    def reorder_bvals_by_direction(self, image, b_values):
        total_bvals = len(b_values)

        # Find unique b-values (excluding 0)
        unique_bvals = torch.unique(b_values[b_values > 0], sorted=True)
        unique_bvals_with_b0 = torch.unique(b_values, sorted=True)
        num_diffusion_bvals = len(unique_bvals)
        num_dirs = Config.ADC_CONFIG["num_dirs"]

        # Verify the structure: 1 b=0 + num_diffusion_bvals * num_dirs = total_bvals
        expected_total = 1 + num_diffusion_bvals * num_dirs
        if total_bvals != expected_total:
            raise ValueError(
                f"B-value structure mismatch: expected {expected_total} b-values "
                f"(1 b=0 + {num_diffusion_bvals} diffusion × {num_dirs} directions), "
                f"but got {total_bvals}"
            )

        # Extract b=0 separately
        b0_image = image[:, 0:1, :, :]

        # Extract and reshape diffusion-weighted images
        dwi_images = image[
            :, 1:, :, :
        ]  # Shape: (slices, num_diffusion_bvals*num_dirs, height, width)

        # Reshape to separate directions: (slices, num_diffusion_bvals, num_dirs, height, width)
        dwi_images_reshaped = dwi_images.view(
            image.shape[0],
            num_diffusion_bvals,
            num_dirs,
            image.shape[2],
            image.shape[3],
        )

        # Use PyTorch operations to reorder by direction
        # Permute to get all x, then all y, then all z: (slices, num_dirs, num_diffusion_bvals, height, width)
        dwi_images_reordered = dwi_images_reshaped.permute(0, 2, 1, 3, 4)

        # Return b0 separately and only diffusion-weighted b-values
        return b0_image, dwi_images_reordered, unique_bvals_with_b0

    def preprocess(self, image, b_values):
        image_padded = self.pad_to_unet_compatible(image)
        image_norm, min_val, max_val = self.normalize_to_b0(image_padded)
        image_reshaped = self.reorder_slices_and_bvals(image_norm)

        # Reorder b-values by direction
        b0_image, dwi_images_reordered, unique_bvals_with_b0 = (
            self.reorder_bvals_by_direction(image_reshaped, b_values)
        )

        return b0_image, dwi_images_reordered, unique_bvals_with_b0, min_val, max_val
