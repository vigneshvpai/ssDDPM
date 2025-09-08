import lightning as L
import glob
import os
import argparse
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.model.SSDDPM import SSDDPM
from src.data.DWIDataLoader import DWIDataLoader
from src.config.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train SSDDPM model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--hpc", action="store_true", help="Run in HPC mode using $TMPDIR/pt_data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hpc:
        data_module = DWIDataLoader(
            train_json=Config.HPC_TRAIN_JSON,
            val_json=Config.HPC_VAL_JSON,
            test_json=Config.HPC_TEST_JSON,
            data_root=Config.HPC_DATA_ROOT,
        )
    else:
        data_module = DWIDataLoader(
            train_json=Config.TRAIN_SPLIT_JSON,
            val_json=Config.VAL_SPLIT_JSON,
            test_json=Config.TEST_SPLIT_JSON,
            data_root=Config.PT_DATA_ROOT,
        )

    model = SSDDPM(
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
    )

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=Config.LOGGER_CONFIG["save_dir"],
        name=Config.LOGGER_CONFIG["name"],
        version=None,  # Auto-increment version
        default_hp_metric=False,
    )

    # Set up CSV logger
    csv_logger = CSVLogger(
        save_dir=Config.LOGGER_CONFIG["save_dir"],
        name=Config.LOGGER_CONFIG["name"],
        version=None,  # Auto-increment version
    )

    # Combine both loggers
    loggers = [tb_logger, csv_logger]

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Config.CHECKPOINT_CONFIG["save_dir"],
            filename=Config.CHECKPOINT_CONFIG["filename"],
            monitor=Config.CHECKPOINT_CONFIG["monitor"],
            mode=Config.CHECKPOINT_CONFIG["mode"],
            every_n_epochs=Config.CHECKPOINT_CONFIG["every_n_epochs"],
            save_top_k=Config.CHECKPOINT_CONFIG["save_top_k"],
        ),
        LearningRateMonitor(logging_interval="epoch"),  # Log LR at every step
    ]

    # Find the latest checkpoint if resuming training
    if args.resume:
        checkpoint_dir = Config.CHECKPOINT_CONFIG["save_dir"]
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        latest_checkpoint = (
            max(checkpoints, key=os.path.getctime) if checkpoints else None
        )
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
    else:
        print("Starting training from scratch")
        latest_checkpoint = None

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        max_epochs=Config.SSDDPM_CONFIG["max_epochs"],
        enable_checkpointing=True,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=True,
        log_every_n_steps=Config.SSDDPM_CONFIG["log_every_n_steps"],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=latest_checkpoint)


if __name__ == "__main__":
    main()
