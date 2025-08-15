import torch
import lightning as L
import diffusers
from .unet_2d import CustomUNet2D


class SSDDPM(L.LightningModule):
    def __init__(self, in_channels=3, out_channels=3, sample_size=32):
        super().__init__()
        # Use custom UNet instead of default diffusers UNet
        self.model = CustomUNet2D(
            in_channels=in_channels, out_channels=out_channels, sample_size=sample_size
        )

    def forward(self, x):
        return self.model(x)
