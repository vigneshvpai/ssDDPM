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


if __name__ == "__main__":
    main()
