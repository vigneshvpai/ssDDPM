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
    def normalize_to_minus_1_1(images: torch.Tensor):
        """
        images: (B, 9, H, W) where 9 = n_bvals
        Normalize each image's b-value curve together
        """
        # Compute min/max per image (across all b-values)
        min_val = images.view(images.shape[0], -1).min(dim=1)[0]  # (B,)
        max_val = images.view(images.shape[0], -1).max(dim=1)[0]  # (B,)

        # Reshape for broadcasting
        min_val = min_val.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        max_val = max_val.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        images_norm = 2 * (images - min_val) / (max_val - min_val + 1e-8) - 1
        return images_norm, min_val, max_val

    @staticmethod
    def normalize_to_b0(image):
        """
        Normalize the image to the b0 value (first b-value).
        Args:
            image (torch.Tensor): The input image tensor of shape (batch, bvalues, width, height).
        Returns:
            torch.Tensor: The image normalized to b0 values.
        """
        # Extract b0 values (first b-value) - shape: (batch, 1, width, height)
        b0_image = image[:, 0:1, :, :]  # Keep dimensions for broadcasting

        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8

        # Normalize each b-value by the corresponding b0 value
        # Broadcasting: (batch, bvalues, width, height) / (batch, 1, width, height)
        normalized_image = image / (b0_image + epsilon)

        return normalized_image, b0_image

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
    def filter_bvals_by_indices(image, indices):
        """
        Filter b-values from the image tensor using specified indices.
        Args:
            image (torch.Tensor): Image tensor of shape (batch_size, bvalues, width, height)
                                 or (bvalues, width, height).
            indices (list or torch.Tensor): List or tensor of indices to keep.
        Returns:
            torch.Tensor: Filtered image tensor with only the specified b-values.
        """
        if image.ndim == 3:
            # Single image: (bvalues, width, height)
            return image[indices, :, :]
        elif image.ndim == 4:
            # Batched image: (batch_size, bvalues, width, height)
            return image[:, indices, :, :]
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {image.ndim}D tensor with shape {image.shape}"
            )

    @staticmethod
    def pad_to_unet_compatible(image, target_shape=None):
        """
        Pad the image tensor with zeros to make height and width match target_shape.
        Args:
            image (torch.Tensor): Image tensor of shape (width, height, bvalues) or
                                 (batch_size, bvalues, width, height).
            target_shape (tuple): (target_width, target_height)
        Returns:
            torch.Tensor: Zero-padded image tensor.
        """
        if target_shape is None:
            target_shape = Config.UNET_COMPATIBLE_SHAPE

        if image.ndim == 3:
            # Single image: (width, height, bvalues)
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

        elif image.ndim == 4:
            # Batched image: (batch_size, bvalues, width, height)
            batch_size, b, w, h = image.shape
            target_w, target_h = target_shape

            pad_h = max(target_h - h, 0)
            pad_w = max(target_w - w, 0)

            # Calculate padding for height and width dimensions
            pad_left_h = pad_h // 2
            pad_right_h = pad_h - pad_left_h
            pad_left_w = pad_w // 2
            pad_right_w = pad_w - pad_left_w

            # For torch.nn.functional.pad, dimensions are padded from right to left
            # For shape (batch_size, bvalues, width, height):
            # - Dimension 3 (height): pad_left_h, pad_right_h
            # - Dimension 2 (width): pad_left_w, pad_right_w
            # - Dimensions 1 (bvalues) and 0 (batch_size): no padding
            pad = (pad_left_h, pad_right_h, pad_left_w, pad_right_w)

        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {image.ndim}D tensor with shape {image.shape}"
            )

        # Note: torch.nn.functional.pad() always creates new memory
        image_padded = torch.nn.functional.pad(image, pad)
        return image_padded

    @staticmethod
    def preprocess(image, b_values):
        # image, min_val, max_val = Preprocess.normalize_to_0_1(image)
        # image = Preprocess.pad_to_unet_compatible(image)
        image = Preprocess.reorder_bvals(image)
        image = Preprocess.filter_bvals_by_indices(image, [2, 5, 7])
        b_values = b_values[[2, 5, 7]]

        return image, b_values
