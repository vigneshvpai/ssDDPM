import lightning as L
import glob
import os
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

    # Find the latest checkpoint
    checkpoint_dir = Config.CHECKPOINT_CONFIG["save_dir"]
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    latest_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        max_epochs=Config.SSDDPM_CONFIG["max_epochs"],
        enable_checkpointing=True,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=latest_checkpoint)


if __name__ == "__main__":
    main()
