import torch
import lightning as L
from diffusers.models import UNet2DModel


class UNetAutoencoder(L.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        lr=1e-3,
        sample_size=128,
        norm_num_groups=8,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.save_hyperparameters(ignore=[])
        self.lr = lr

        # Four blocks, each with two layers, channels: 64, 128, 128, 256
        # Two out of four block transitions include spatial self-attention
        # Down: [64, 128, 128, 256], Up: [256, 128, 128, 64]
        # Attention in 2nd and 4th blocks (index 1 and 3)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=[64, 128, 128, 256],
            down_block_types=(
                "DownBlock2D",  # 64
                "AttnDownBlock2D",  # 128 (attention)
                "DownBlock2D",  # 128
                "AttnDownBlock2D",  # 256 (attention)
            ),
            up_block_types=(
                "AttnUpBlock2D",  # 256 (attention)
                "UpBlock2D",  # 128
                "AttnUpBlock2D",  # 128 (attention)
                "UpBlock2D",  # 64
            ),
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        return self.unet(x).sample

    def training_step(self, batch, batch_idx):
        # Self-supervised: input is the target
        if isinstance(batch, dict):
            x = batch["image"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        out = self(x)
        loss = torch.nn.functional.mse_loss(out, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["image"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        out = self(x)
        loss = torch.nn.functional.mse_loss(out, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
