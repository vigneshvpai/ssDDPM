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
        n_slices = slices_times_bvals // Config.DWI_CONFIG["n_bvals"]

        slices = []
        for i in range(n_slices):
            slice_indices = torch.arange(i, slices_times_bvals, n_slices)
            slice_data = image[slice_indices]
            slices.append(slice_data)

        image = torch.stack(slices, dim=0)

        # Permute back to (width, height, slices, bvalues)
        image = image.permute(3, 2, 0, 1)

        return image

    @staticmethod
    def unpad_from_unet_compatible(image, original_shape=None):
        """
        Reverse the pad_to_unet_compatible operation.
        Removes padding to restore original dimensions.
        Args:
            image (torch.Tensor): Image tensor of shape (batch_size, bvalues, width, height) or (bvalues, width, height).
            original_shape (tuple): (original_width, original_height) - defaults to EXPECTED_SHAPE
        Returns:
            torch.Tensor: Unpadded image tensor.
        """
        if original_shape is None:
            original_shape = (
                Config.EXPECTED_SHAPE[0],  # width = 108
                Config.EXPECTED_SHAPE[1],  # height = 134
            )  # (108, 134)

        # Handle batch dimension
        has_batch = image.ndim == 4
        if has_batch:
            batch_size, b, w, h = image.shape  # (batch_size, bvalues, width, height)
        else:
            b, w, h = image.shape  # (bvalues, width, height)

        original_w, original_h = original_shape

        # Calculate padding that was added
        pad_w = w - original_w
        pad_h = h - original_h

        if pad_w < 0 or pad_h < 0:
            raise ValueError(
                f"Image is smaller than original shape: {image.shape} vs {original_shape}"
            )

        # Calculate the padding that was applied (same logic as pad_to_unet_compatible)
        pad_left_w = pad_w // 2
        pad_right_w = pad_w - pad_left_w
        pad_left_h = pad_h // 2
        pad_right_h = pad_h - pad_left_h

        # Remove padding by slicing
        if has_batch:
            # For (batch_size, bvalues, width, height)
            image = image[
                :, :, pad_left_w : w - pad_right_w, pad_left_h : h - pad_right_h
            ]
        else:
            # For (bvalues, width, height)
            image = image[:, pad_left_w : w - pad_right_w, pad_left_h : h - pad_right_h]

        return image

    @staticmethod
    def denormalize_from_0_1(image, original_min=None, original_max=None):
        """
        Reverse normalization to b0 using original min and max values.
        Args:
            image (torch.Tensor): Normalized image tensor of shape (batch_size, bvalues, height, width).
            original_min (float, torch.Tensor, or dict): Minimum value(s) used during normalization.
                                                       Can be scalar, tensor, or dict with 'min_val' key.
            original_max (float, torch.Tensor, or dict): Maximum value(s) used during normalization.
                                                       Can be scalar, tensor, or dict with 'max_val' key.
        Returns:
            torch.Tensor: Denormalized image tensor.
        """
        if original_min is None or original_max is None:
            raise ValueError(
                "original_min and original_max must be provided for denormalization."
            )

        # Handle dict input (extract min_val and max_val)
        if isinstance(original_min, dict) and "min_val" in original_min:
            original_min = original_min["min_val"]
        if isinstance(original_max, dict) and "max_val" in original_max:
            original_max = original_max["max_val"]

        # Ensure original_min and original_max are tensors
        if not isinstance(original_min, torch.Tensor):
            original_min = torch.tensor(original_min)
        if not isinstance(original_max, torch.Tensor):
            original_max = torch.tensor(original_max)

        # Handle batch dimension
        if image.ndim == 4:  # (batch_size, bvalues, height, width)
            batch_size = image.shape[0]

            # Reshape min/max to broadcast across batch and spatial dimensions
            # original_min/max shape: (batch_size,) -> (batch_size, 1, 1, 1)
            if original_min.shape == (batch_size,):
                original_min = original_min.view(batch_size, 1, 1, 1)
            elif original_min.numel() == 1:
                # Single value for entire batch
                original_min = original_min.view(1, 1, 1, 1)
            else:
                raise ValueError(
                    f"original_min shape {original_min.shape} incompatible with batch_size {batch_size}"
                )

            if original_max.shape == (batch_size,):
                original_max = original_max.view(batch_size, 1, 1, 1)
            elif original_max.numel() == 1:
                # Single value for entire batch
                original_max = original_max.view(1, 1, 1, 1)
            else:
                raise ValueError(
                    f"original_max shape {original_max.shape} incompatible with batch_size {batch_size}"
                )

        # Undo normalization: x = x_norm * (max - min) + min
        return image * (original_max - original_min) + original_min

    @staticmethod
    def denormalize_from_minus_1_1(image: torch.Tensor, min_val, max_val):
        """
        Denormalize an image from [-1, 1] back to [min_val, max_val].

        Args:
            image (torch.Tensor): Normalized image in [-1, 1].
            min_val (float): Original min.
            max_val (float): Original max.
        Returns:
            torch.Tensor: Denormalized image.
        """
        return ((image + 1) / 2) * (max_val - min_val) + min_val
