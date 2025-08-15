import torch
import torch.nn as nn
import lightning as L
import diffusers
from diffusers.models.unet_2d import UNet2DModel


class SSDDPM(L.LightningModule):
    def __init__(
        self, in_channels=3, out_channels=3, sample_size=32, num_timesteps=1000
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Use a simple UNet from the diffusers library as our backbone
        # Enable time embedding by setting time_embedding_type
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,  # Two layers per block as in our spec
            block_out_channels=(64, 128, 128, 256),  # Our channel configuration
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # Attention at transition 1
                "AttnDownBlock2D",  # Attention at transition 2
            ),
            up_block_types=(
                "AttnUpBlock2D",  # Attention at transition 2
                "AttnUpBlock2D",  # Attention at transition 1
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",  # Use a standard mid block
            time_embedding_type="positional",  # Enable time embedding
            time_embedding_dim=128,  # Dimension of time embedding
            time_embedding_act_fn="silu",  # Activation function for time embedding
        )

    def sample_timesteps(self, batch_size, device=None):
        """
        Sample timesteps from Uniform({1, ..., T})

        Args:
            batch_size: Number of timesteps to sample
            device: Device to place the tensor on (default: same as model)

        Returns:
            timesteps: Tensor of shape (batch_size,) with values from {1, ..., T}
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from Uniform({1, ..., T})
        # Note: We use 1 to T (not 0 to T-1) as per the algorithm
        timesteps = torch.randint(
            low=1,
            high=self.num_timesteps + 1,  # +1 because randint high is exclusive
            size=(batch_size,),
            device=device,
        )

        return timesteps

    def sample_noise(self, shape, device=None):
        """
        Sample noise from N(0, I) - standard normal distribution

        Args:
            shape: Shape of the noise tensor (e.g., (batch_size, channels, height, width))
            device: Device to place the tensor on (default: same as model)

        Returns:
            noise: Tensor of shape 'shape' sampled from N(0, I)
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal distribution N(0, I)
        noise = torch.randn(shape, device=device)

        return noise

    def forward(self, x, timesteps=None):
        """
        Forward pass through the DDPM model

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Timestep tensor of shape (batch_size,) indicating noise level.
                      If None, will sample timesteps automatically.
        """
        batch_size = x.shape[0]

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size, device=x.device)

        # Forward pass through our UNet model with time conditioning
        return self.model(x, timesteps=timesteps)
