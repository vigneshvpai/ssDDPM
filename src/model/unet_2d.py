import torch
import torch.nn as nn
from diffusers.models.attention import Attention

# This is a custom implementation of a 2D UNet architecture.
# UNet is widely used for image-to-image tasks (e.g., segmentation, denoising, diffusion models).
# This version uses four blocks, each with two convolutional layers, and spatial self-attention at certain transitions.


class CustomUNet2D(nn.Module):
    """
    Custom 2D UNet with specific architecture:
    - Four blocks, each with two layers
    - Channel configurations: 64, 128, 128, 256
    - Spatial self-attention at two block transitions
    """

    def __init__(self, in_channels=3, out_channels=3, sample_size=32):
        super().__init__()

        # Channel configurations for each block in the encoder/decoder
        self.channels = [64, 128, 128, 256]

        # Initial convolution to map input channels to the first block's channels
        self.conv_in = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        # Downsampling blocks (encoder path)
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            # Use attention at transitions 1 and 2 (i==1 or i==2)
            block = DownBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                use_attention=(i == 1 or i == 2),  # Attention at transitions 1 and 2
            )
            self.down_blocks.append(block)

        # Middle block (bottleneck), always uses attention
        self.middle_block = MiddleBlock(channels=self.channels[-1], use_attention=True)

        # Upsampling blocks (decoder path)
        self.up_blocks = nn.ModuleList()
        # Iterate backwards through the channels for upsampling
        for i in range(len(self.channels) - 1, 0, -1):
            # Use attention at transitions 2 and 1 (i==2 or i==1)
            block = UpBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i - 1],
                use_attention=(i == 2 or i == 1),  # Attention at transitions 2 and 1
            )
            self.up_blocks.append(block)

        # Final convolution to map back to output channels
        self.conv_out = nn.Conv2d(
            self.channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv_in(x)

        # Downsampling path with skip connections
        skip_connections = []
        for block in self.down_blocks:
            x, skip = block(x)  # Each block returns (downsampled_x, skip_x)
            skip_connections.append(skip)

        # Middle block (bottleneck)
        x = self.middle_block(x)

        # Upsampling path with skip connections
        for i, block in enumerate(self.up_blocks):
            # Retrieve corresponding skip connection in reverse order
            skip = skip_connections[-(i + 1)]
            x = block(x, skip)

        # Final convolution to produce output
        x = self.conv_out(x)

        return x


class DownBlock(nn.Module):
    """Downsampling block with two layers and optional attention"""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()

        # First convolutional layer: changes channel size and applies normalization and activation
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Second convolutional layer: keeps channel size, normalization, activation
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Optional spatial self-attention after convolutions
        self.attention = Attention(out_channels) if use_attention else None

        # Downsampling operation (halves spatial size)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        # First convolutional layer
        x = self.layer1(x)

        # Second convolutional layer
        x = self.layer2(x)

        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        # Store output for skip connection (before downsampling)
        skip = x

        # Downsample spatially
        x = self.downsample(x)

        # Return downsampled output and skip connection
        return x, skip


class MiddleBlock(nn.Module):
    """Middle block with attention"""

    def __init__(self, channels, use_attention=True):
        super().__init__()

        # First convolutional layer (keeps channel size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Second convolutional layer (keeps channel size)
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Optional attention (usually always enabled in bottleneck)
        self.attention = Attention(channels) if use_attention else None

    def forward(self, x):
        # Two convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        return x


class UpBlock(nn.Module):
    """Upsampling block with two layers and optional attention"""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()

        # Upsampling operation (doubles spatial size, keeps channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        # First convolutional layer:
        # - Input channels are doubled due to concatenation with skip connection
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, out_channels, kernel_size=3, padding=1
            ),  # *2 for skip connection
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Optional spatial self-attention after convolutions
        self.attention = Attention(out_channels) if use_attention else None

    def forward(self, x, skip):
        # Upsample input (from previous layer)
        x = self.upsample(x)

        # Concatenate with skip connection from encoder (along channel dimension)
        x = torch.cat([x, skip], dim=1)

        # First convolutional layer
        x = self.layer1(x)

        # Second convolutional layer
        x = self.layer2(x)

        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        return x
