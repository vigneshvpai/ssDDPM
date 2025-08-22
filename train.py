import lightning as L
from src.model.SSDDPM import SSDDPM
from src.data.DWIDataLoader import DWIDataLoader


def main():
    # Instantiate the data module
    data_module = DWIDataLoader()

    # Instantiate the SSDDPM model
    model = SSDDPM(in_channels=625, out_channels=1)

    # Set up the trainer for 5 epochs
    trainer = L.Trainer(max_epochs=1)

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
