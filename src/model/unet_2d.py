import torch
import torch.nn as nn
from diffusers.models.attention import Attention


class CustomUNet2D(nn.Module):
    """
    Custom 2D UNet with specific architecture:
    - Four blocks, each with two layers
    - Channel configurations: 64, 128, 128, 256
    - Spatial self-attention at two block transitions
    """

    def __init__(self, in_channels=3, out_channels=3, sample_size=32):
        super().__init__()

        # Channel configurations for each block
        self.channels = [64, 128, 128, 256]

        # Initial convolution
        self.conv_in = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            block = DownBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                use_attention=(i == 1 or i == 2),  # Attention at transitions 1 and 2
            )
            self.down_blocks.append(block)

        # Middle block
        self.middle_block = MiddleBlock(channels=self.channels[-1], use_attention=True)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1, 0, -1):
            block = UpBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i - 1],
                use_attention=(i == 2 or i == 1),  # Attention at transitions 2 and 1
            )
            self.up_blocks.append(block)

        # Final convolution
        self.conv_out = nn.Conv2d(
            self.channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv_in(x)

        # Downsampling path with skip connections
        skip_connections = []
        for block in self.down_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Middle block
        x = self.middle_block(x)

        # Upsampling path with skip connections
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]
            x = block(x, skip)

        # Final convolution
        x = self.conv_out(x)

        return x


class DownBlock(nn.Module):
    """Downsampling block with two layers and optional attention"""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()

        # Two layers as specified
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial self-attention if specified
        self.attention = Attention(out_channels) if use_attention else None

        # Downsampling
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        # First layer
        x = self.layer1(x)

        # Second layer
        x = self.layer2(x)

        # Attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        # Store skip connection
        skip = x

        # Downsample
        x = self.downsample(x)

        return x, skip


class MiddleBlock(nn.Module):
    """Middle block with attention"""

    def __init__(self, channels, use_attention=True):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.attention = Attention(channels) if use_attention else None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        if self.attention is not None:
            x = self.attention(x)

        return x


class UpBlock(nn.Module):
    """Upsampling block with two layers and optional attention"""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()

        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        # Two layers as specified
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, out_channels, kernel_size=3, padding=1
            ),  # *2 for skip connection
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial self-attention if specified
        self.attention = Attention(out_channels) if use_attention else None

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # First layer
        x = self.layer1(x)

        # Second layer
        x = self.layer2(x)

        # Attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        return x
