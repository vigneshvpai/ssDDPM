import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self):
        self.num_dirs = Config.DWI_CONFIG["num_dirs"]
        self.n_bvals = Config.DWI_CONFIG["n_bvals"]

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

    def reorder_bvals(self, image):
        # Expecting image shape: (width, height, bvalues)
        if image.ndim != 3:
            raise ValueError(
                f"Expected image of shape (width, height, bvalues), got {image.shape}"
            )
        # Permute to (bvalues, height, width)
        image = image.permute(2, 0, 1)

        return image

    def pad_to_unet_compatible(self, image, target_shape=None):
        """
        Pad the image tensor with zeros to make width and height match target_shape.
        Args:
            image (torch.Tensor): Image tensor of shape (width, height, bvals).
            target_shape (tuple): (target_height, target_width)
        Returns:
            torch.Tensor: Zero-padded image tensor.
        """
        if target_shape is None:
            target_shape = Config.UNET_COMPATIBLE_SHAPE

        # image shape: (width, height, slices, bvals)
        w, h, b = image.shape
        target_h, target_w = target_shape

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # Calculate padding for width and height dimensions
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h

        # For padding width and height (first two dimensions)
        pad = (0, 0, pad_left_h, pad_right_h, pad_left_w, pad_right_w)
        # Note: torch.nn.functional.pad() always creates new memory
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    def preprocess(self, image):
        image = self.pad_to_unet_compatible(image)
        image, min_val, max_val = self.normalize_to_b0(image)  # Now in-place
        image = self.reorder_bvals(image)

        return image, min_val, max_val
