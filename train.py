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
        save_dir=Config.LOGGING_CONFIG["save_dir"],
        name=Config.LOGGING_CONFIG["name"],
        version=None,  # Auto-increment version
        log_graph=Config.LOGGING_CONFIG["log_graph"],
        default_hp_metric=False,
        flush_secs=10,  # Flush logs every 10 seconds
        log_hyperparameters=Config.LOGGING_CONFIG["log_hyperparameters"],
    )

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Config.CHECKPOINT_CONFIG["save_dir"],
            filename=Config.CHECKPOINT_CONFIG["filename"],
            save_top_k=Config.CHECKPOINT_CONFIG["save_top_k"],
            monitor=Config.CHECKPOINT_CONFIG["monitor"],
            mode=Config.CHECKPOINT_CONFIG["mode"],
            every_n_epochs=Config.CHECKPOINT_CONFIG["every_n_epochs"],
        ),
        LearningRateMonitor(logging_interval="step"),  # Log LR at every step
    ]

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        enable_checkpointing=True,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=Config.LOGGING_CONFIG["log_every_n_steps"],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
