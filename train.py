import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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

    # Set up TensorBoard logger with more options
    logger = TensorBoardLogger(
        save_dir=Config.LOGGER_CONFIG["save_dir"],
        name=Config.LOGGER_CONFIG["name"],
        version=None,  # Auto-increment version
        log_graph=Config.LOGGER_CONFIG["log_graph"],
        default_hp_metric=False,
    )

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Config.CHECKPOINT_CONFIG["save_dir"],
            filename=Config.CHECKPOINT_CONFIG["filename"],
            monitor=Config.CHECKPOINT_CONFIG["monitor"],
            mode=Config.CHECKPOINT_CONFIG["mode"],
            every_n_epochs=Config.CHECKPOINT_CONFIG["every_n_epochs"],
        ),
        LearningRateMonitor(logging_interval="epoch"),  # Log LR at every step
    ]

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        max_epochs=Config.SSDDPM_CONFIG["max_epochs"],
        enable_checkpointing=True,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=Config.LOGGER_CONFIG["log_every_n_steps"],
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
