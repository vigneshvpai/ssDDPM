import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from src.model.SSDDPM import SSDDPM
from src.data.DWIDataLoader import DWIDataLoader
from src.config.config import Config


def main():
    # Instantiate the data module
    data_module = DWIDataLoader()

    model = SSDDPM(
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
    )

    # Set up the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="ssddpm")

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


def train_one_batch():
    """
    Trains the model for a single batch and returns the loss.
    """
    data_module = DWIDataLoader()
    data_module.setup()
    model = SSDDPM(
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
    )
    model.train()
    dataloader = data_module.train_dataloader()
    batch = next(iter(dataloader))
    optimizer = model.configure_optimizers()["optimizer"]

    optimizer.zero_grad()
    loss = model.training_step(batch)
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    train_one_batch()
