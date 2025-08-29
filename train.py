import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
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

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="ssddpm-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,  # Keep top 3 best models
                monitor="val_loss",  # Monitor validation loss
                mode="min",  # Lower is better
                every_n_epochs=1,  # Save every epoch
            )
        ],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
