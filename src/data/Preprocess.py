import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self, pt_data_root=Config.PT_DATA_ROOT):
        self.pt_data_root = pt_data_root

    def normalize_to_b0(self, sample):
        """
        Normalize the image to the b=0 image (assumed to be at last dimension index 0),
        and scale the result to the 0-1 range based on the b=0 image.
        Args:
            sample (dict): A sample dict with key 'image'.
        Returns:
            dict: The sample with 'image' normalized to b0 and scaled to 0-1.
        """
        image = sample["image"]
        # Assume image shape is [..., N_bvals], and b=0 is at index 0 of last dim
        b0 = image[..., 0]
        epsilon = 1e-8
        # Expand b0 to match image shape for broadcasting
        b0_expanded = b0.unsqueeze(-1)
        image_norm = image / (b0_expanded + epsilon)
        # Now scale to 0-1 range based on b0 min/max
        b0_min = b0.min()
        b0_max = b0.max()
        # Avoid division by zero if b0 is constant
        scale = (b0_max - b0_min) if (b0_max - b0_min) > 0 else 1.0
        image_norm = (image_norm - b0_min) / scale
        sample["image"] = image_norm
        return sample

    def reshape_bvals_to_channels(self, sample):
        """
        Reshape the image tensor from (width, height, slices, bvalues) to (bvalues, slices, height, width),
        so that bvalues become the channel dimension, compatible with PyTorch training.
        Args:
            sample (dict): A sample dict with key 'image'.
        Returns:
            torch.Tensor: The reshaped image tensor.
        """
        image = sample["image"]
        # Expecting image shape: (width, height, slices, bvalues)
        if image.ndim != 4:
            raise ValueError(
                f"Expected image of shape (width, height, slices, bvalues), got {image.shape}"
            )
        # Permute to (bvalues, slices, height, width)
        image = image.permute(3, 2, 1, 0)
        return image

    def pad_to_unet_compatible(self, image, target_shape=None):
        """
        Pad the image tensor with zeros to make width and height match target_shape.
        Args:
            image (torch.Tensor): Image tensor of shape (bvalues, slices, height, width).
            target_shape (tuple): (target_height, target_width)
        Returns:
            torch.Tensor: Zero-padded image tensor.
        """
        if target_shape is None:
            target_shape = Config.UNET_COMPATIBLE_SHAPE

        # image shape: (bvalues, slices, height, width)
        b, s, h, w = image.shape
        target_h, target_w = target_shape

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # Padding: (left, right, top, bottom) for last two dims
        # torch.nn.functional.pad uses (last dim, second last dim, ...)
        # So pad = (left_w, right_w, left_h, right_h)
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h

        # Pad order: (left_w, right_w, left_h, right_h)
        pad = (pad_left_w, pad_right_w, pad_left_h, pad_right_h)
        # torch.nn.functional.pad expects pad for last two dims, so input must be (N, C, H, W) or similar
        # Here, image is (bvalues, slices, height, width), so pad applies to (width, height)
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    def preprocess(self, sample):
        """
        Preprocess a sample by normalizing to b0, reshaping the image, and padding to U-Net compatible size.
        Args:
            sample (dict): A sample dict with key 'image'.
        Returns:
            torch.Tensor: The preprocessed and padded image tensor.
        """
        sample = self.normalize_to_b0(sample)
        image = self.reshape_bvals_to_channels(sample)
        image_padded = self.pad_to_unet_compatible(image)
        return image_padded
