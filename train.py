import lightning as L
import glob
import os
import torch
import argparse
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.model.SSDDPM import SSDDPM
from src.data.DWIDataLoader import DWIDataLoader
from src.config.config import Config

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SSDDPM model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--hpc", action="store_true", help="Run in HPC mode using $TMPDIR/pt_data"
    )
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    print(Config.summary())

    # Get SLURM job id if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id:
        run_name = f"{args.exp_name}_slurm{slurm_job_id}"
    else:
        run_name = args.exp_name

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
            data_root=Config.PT_DATA_ROOT_SLICEWISE,
        )

    model = SSDDPM(
        in_channels=Config.SSDDPM_CONFIG["in_channels"],
        out_channels=Config.SSDDPM_CONFIG["out_channels"],
        run_name=run_name,
    )

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=Config.LOGGER_CONFIG["save_dir"],
        name=run_name,
        version=None,  # Auto-increment version
        default_hp_metric=False,
    )

    # Set up CSV logger
    csv_logger = CSVLogger(
        save_dir=Config.LOGGER_CONFIG["save_dir"],
        name=run_name,
        version=None,  # Auto-increment version
    )

    # Combine both loggers
    loggers = [tb_logger, csv_logger]

    # Set up callbacks
    checkpoint_dir = os.path.join(Config.CHECKPOINT_CONFIG["save_dir"], run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=Config.CHECKPOINT_CONFIG["filename"],
            monitor=Config.CHECKPOINT_CONFIG["monitor"],
            mode=Config.CHECKPOINT_CONFIG["mode"],
            every_n_epochs=Config.CHECKPOINT_CONFIG["every_n_epochs"],
            save_top_k=Config.CHECKPOINT_CONFIG["save_top_k"],
        ),
        LearningRateMonitor(logging_interval="epoch"),  # Log LR at every step
    ]

    # Determine checkpoint path for resuming training
    latest_checkpoint = None
    if args.checkpoint_path:
        # Use the specified checkpoint path
        if os.path.exists(args.checkpoint_path):
            latest_checkpoint = args.checkpoint_path
            print(f"Resuming training from specified checkpoint: {latest_checkpoint}")
        else:
            raise FileNotFoundError(
                f"Checkpoint file not found: {args.checkpoint_path}"
            )
    elif args.resume:
        # Find the latest checkpoint in the checkpoint directory
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        latest_checkpoint = (
            max(checkpoints, key=os.path.getctime) if checkpoints else None
        )
        if latest_checkpoint:
            print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoints found in checkpoint directory, starting from scratch")
    else:
        print("Starting training from scratch")

    # Set up the trainer using max_epochs from config and the logger
    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=Config.SSDDPM_CONFIG["max_epochs"],
        enable_checkpointing=True,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=Config.LOGGER_CONFIG["enable_progress_bar"],
        enable_model_summary=True,
        log_every_n_steps=Config.SSDDPM_CONFIG["log_every_n_steps"],
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=latest_checkpoint)

    # Save the final model checkpoint at the end of training
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Final model checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
