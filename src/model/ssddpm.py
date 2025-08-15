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

    def forward(self, x, timesteps):
        """
        Forward pass through the DDPM model

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Timestep tensor of shape (batch_size,) indicating noise level
        """
        # Forward pass through our UNet model with time conditioning
        return self.model(x, timesteps=timesteps)
