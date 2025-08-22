import lightning as L
from src.model.dummy_model import BasicUNet
from src.data.DWIDataLoader import DWIDataLoader


def main():
    # Instantiate the data module
    data_module = DWIDataLoader()

    # Instantiate the model
    model = BasicUNet(in_channels=625, out_channels=625)

    # Set up the trainer for 5 epochs
    trainer = L.Trainer(max_epochs=5)

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
