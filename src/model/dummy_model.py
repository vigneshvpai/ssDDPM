import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F


class BasicUNet(L.LightningModule):
    """
    A minimal UNet for testing DataLoader.
    The model learns to map input images to random noise (Gaussian) as ground truth.
    """

    def __init__(self, in_channels=1, out_channels=1, lr=1e-3, img_size=128):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final(d1)
        return out

    def training_step(self, batch, batch_idx):
        # Accepts batch as dict, tuple/list, or tensor
        if isinstance(batch, dict):
            x = batch["image"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # Generate random noise as ground truth
        noise = torch.randn_like(x)
        out = self(x)
        loss = F.mse_loss(out, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["image"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        noise = torch.randn_like(x)
        out = self(x)
        loss = F.mse_loss(out, noise)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
