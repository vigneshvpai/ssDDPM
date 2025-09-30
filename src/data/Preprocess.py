import os
import torch
from src.config.config import Config


class Preprocess:
    @staticmethod
    def normalize_to_0_1(image):
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

        normalized_image = (image - min_val) / scale

        return normalized_image, min_val, max_val

    @staticmethod
    def reorder_bvals(image):
        # Expecting image shape: (width, height, bvalues)
        if image.ndim != 3:
            raise ValueError(
                f"Expected image of shape (width, height, bvalues), got {image.shape}"
            )
        # Permute to (bvalues, width, height)
        image = image.permute(2, 0, 1)

        return image

    @staticmethod
    def pad_to_unet_compatible(image, target_shape=None):
        """
        Pad the image tensor with zeros to make height and width match target_shape.
        Args:
            image (torch.Tensor): Image tensor of shape (width, height, bvalues).
            target_shape (tuple): (target_width, target_height)
        Returns:
            torch.Tensor: Zero-padded image tensor.
        """
        if target_shape is None:
            target_shape = Config.UNET_COMPATIBLE_SHAPE

        # image shape: (width, height, bvalues)
        w, h, b = image.shape
        target_w, target_h = target_shape

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # Calculate padding for height and width dimensions
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w

        # For torch.nn.functional.pad, dimensions are padded from right to left
        # For shape (width, height, bvalues):
        # - Dimension 2 (bvalues): no padding
        # - Dimension 1 (height): pad_left_h, pad_right_h
        # - Dimension 0 (width): pad_left_w, pad_right_w
        pad = (0, 0, pad_left_h, pad_right_h, pad_left_w, pad_right_w)
        # Note: torch.nn.functional.pad() always creates new memory
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    @staticmethod
    def preprocess(image):
        image, min_val, max_val = Preprocess.normalize_to_0_1(image)
        image = Preprocess.pad_to_unet_compatible(image)
        image = Preprocess.reorder_bvals(image)

        return image, min_val, max_val
