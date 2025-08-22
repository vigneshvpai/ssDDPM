import lightning as L
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

    # Set up the trainer using max_epochs from config
    trainer = L.Trainer(max_epochs=Config.MAX_EPOCHS)

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
