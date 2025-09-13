import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self):
        self.num_dirs = Config.ADC_CONFIG["num_dirs"]
        self.n_bvals = Config.ADC_CONFIG["n_bvals"]

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

        # In-place operations
        image.sub_(min_val)  # image = image - min_val
        image.div_(scale)  # image = image / scale

        return image, min_val, max_val

    def reorder_slices_and_bvals(self, image):
        # Expecting image shape: (width, height, slices, bvalues)
        if image.ndim != 4:
            raise ValueError(
                f"Expected image of shape (width, height, slices, bvalues), got {image.shape}"
            )
        # Permute to (slices, bvalues, height, width)
        image = image.permute(2, 3, 1, 0)

        # Split bvalues into num_dirs and n_bvals
        image = self.split_bvals_to_dirs_and_bvals(image)

        return image

    def split_bvals_to_dirs_and_bvals(self, image):
        """
        Split the bvalues dimension to separate num_dirs and n_bvals.

        Args:
            image (torch.Tensor): Image tensor of shape (slices, bvalues, height, width)
                                  where bvalues = 1 + num_dirs * (n_bvals - 1)

        Returns:
            torch.Tensor: Image tensor of shape (slices, num_dirs, n_bvals, height, width)
        """
        if image.ndim != 4:
            raise ValueError(
                f"Expected image of shape (slices, bvalues, height, width), got {image.shape}"
            )

        slices, bvalues, height, width = image.shape

        # Expected structure: b0 + num_dirs * (n_bvals - 1)
        expected_bvalues = 1 + self.num_dirs * (self.n_bvals - 1)
        if bvalues != expected_bvalues:
            raise ValueError(
                f"Expected {expected_bvalues} b-values (1 b0 + {self.num_dirs} dirs × {self.n_bvals-1} bvals), "
                f"got {bvalues}"
            )

        # Split b0 from the rest
        b0_image = image[:, 0:1, :, :]  # Shape: (slices, 1, height, width)
        non_b0_image = image[
            :, 1:, :, :
        ]  # Shape: (slices, num_dirs * (n_bvals - 1), height, width)

        # Reshape non-b0 values: (slices, num_dirs * (n_bvals - 1), height, width)
        # -> (slices, num_dirs, n_bvals - 1, height, width)
        non_b0_reshaped = non_b0_image.view(
            slices, self.num_dirs, self.n_bvals - 1, height, width
        )

        # Concatenate b0 with each direction
        # b0 needs to be repeated for each direction: (slices, 1, height, width) -> (slices, num_dirs, 1, height, width)
        b0_repeated = b0_image.unsqueeze(1).expand(
            slices, self.num_dirs, 1, height, width
        )

        # Concatenate b0 with each direction's b-values
        # Shape: (slices, num_dirs, n_bvals, height, width)
        result = torch.cat([b0_repeated, non_b0_reshaped], dim=2)

        return result

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
        # Note: torch.nn.functional.pad() always creates new memory
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    def preprocess(self, image):
        # Work on a copy to avoid modifying the original
        image = image.clone()

        image = self.pad_to_unet_compatible(image)
        image, min_val, max_val = self.normalize_to_b0(image)  # Now in-place
        image = self.reorder_slices_and_bvals(image)

        # Clear GPU cache after preprocessing if on GPU
        if image.is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image, min_val, max_val
