import torch
from src.config.config import Config


class Postprocess:
    @staticmethod
    def unflatten_slices_and_bvals(image):
        """
        Reverse the flatten_slices_and_bvals operation.
        Converts image from (slices * bvalues, height, width) back to (width, height, slices, bvalues).
        Args:
            image (torch.Tensor): The input image tensor of shape (slices * bvalues, height, width).
        Returns:
            torch.Tensor: The reshaped image tensor with shape (width, height, slices, bvalues).
        """
        # Expecting image shape: (slices * bvalues, height, width)
        if image.ndim != 3:
            raise ValueError(
                f"Expected image of shape (slices * bvalues, height, width), got {image.shape}"
            )

        # Get dimensions
        slices_times_bvals, height, width = image.shape
        n_slices = 25  # From config
        n_bvals = Config.ADC_CONFIG["n_bvals"]  # 25

        # Reshape to (slices, bvalues, height, width)
        image = image.reshape(n_slices, n_bvals, height, width)

        # Permute back to (width, height, slices, bvalues)
        image = image.permute(3, 2, 0, 1)

        return image

    @staticmethod
    def unpad_from_unet_compatible(image, original_shape=None):
        """
        Reverse the pad_to_unet_compatible operation.
        Removes padding to restore original dimensions.
        Args:
            image (torch.Tensor): Image tensor of shape (width, height, slices, bvals).
            original_shape (tuple): (original_width, original_height) - defaults to EXPECTED_SHAPE
        Returns:
            torch.Tensor: Unpadded image tensor.
        """
        if original_shape is None:
            original_shape = (
                Config.EXPECTED_SHAPE[0],
                Config.EXPECTED_SHAPE[1],
            )  # (108, 134)

        # image shape: (width, height, slices, bvals)
        w, h, s, b = image.shape
        original_w, original_h = original_shape

        # Calculate padding that was added
        pad_w = w - original_w
        pad_h = h - original_h

        if pad_w < 0 or pad_h < 0:
            raise ValueError(
                f"Image is smaller than original shape: {image.shape} vs {original_shape}"
            )

        # Calculate the padding that was applied
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h

        # Remove padding by slicing
        # Remove from width dimension (first dimension)
        image = image[pad_left_w : w - pad_right_w, :, :, :]
        # Remove from height dimension (second dimension)
        image = image[:, pad_left_h : h - pad_right_h, :, :]

        return image

    @staticmethod
    def denormalize_from_b0(image, original_min=None, original_max=None):
        """
        Reverse normalization to b0 using original min and max values.
        Args:
            image (torch.Tensor): Normalized image tensor.
            original_min (float or torch.Tensor): Minimum value used during normalization.
            original_max (float or torch.Tensor): Maximum value used during normalization.
        Returns:
            torch.Tensor: Denormalized image tensor.
        """
        if original_min is None or original_max is None:
            raise ValueError(
                "original_min and original_max must be provided for denormalization."
            )

        # Undo normalization: x = x_norm * (max - min) + min
        return image * (original_max - original_min) + original_min
