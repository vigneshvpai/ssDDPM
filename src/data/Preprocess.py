import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self, pt_data_root=Config.PT_DATA_ROOT):
        self.pt_data_root = pt_data_root

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
        return image_norm

    def reshape_bvals_to_channels(self, image):
        """
        Reshape the image tensor from (width, height, slices, bvalues) to (slices, bvalues, height, width),
        so that slices become the batch dimension and bvalues become the channel dimension.
        Args:
            image (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The reshaped image tensor.
        """
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

    def preprocess(self, image):
        """
        Preprocess a sample by normalizing to b0, reshaping the image, and padding to U-Net compatible size.
        Args:
            sample (dict): A sample dict with key 'image'.
        Returns:
            torch.Tensor: The preprocessed and padded image tensor.
        """
        image_padded = self.pad_to_unet_compatible(image)
        image_norm = self.normalize_to_b0(image_padded)
        image_reshaped = self.reshape_bvals_to_channels(image_norm)

        return image_reshaped
