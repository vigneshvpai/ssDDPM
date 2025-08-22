import torch
import torch.nn as nn
import lightning as L


class DummyEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        return self.encoder(x)


class DummyDecoder(nn.Module):
    def __init__(self, out_channels, latent_dim=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 16 * 8 * 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.decoder(x)
        return x


class DummyAutoencoder(L.LightningModule):
    def __init__(self, in_channels, out_channels=None, latent_dim=16, lr=1e-3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.encoder = DummyEncoder(in_channels, latent_dim)
        self.decoder = DummyDecoder(out_channels, latent_dim)
        self.lr = lr
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def training_step(self, batch, batch_idx):
        # Assume batch is a dict or tuple: (input, target)
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch.get("target", x)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else batch[0]
        else:
            x = batch
            y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch.get("target", x)
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else batch[0]
        else:
            x = batch
            y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
