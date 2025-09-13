import os
import pprint


class Config:
    # -------------------------
    # Path and Environment Configs
    # -------------------------
    ORIGINAL_DATA_ROOT = "/home/vault/mfdp/mfdp118h/data"
    PT_DATA_ROOT = "/home/vault/mfdp/mfdp118h/pt_data"
    TMPDIR = os.environ.get("TMPDIR", "")
    HPC_DATA_ROOT = os.path.join(TMPDIR, "pt_data")

    TRAIN_JSON = "train.json"
    VAL_JSON = "val.json"
    TEST_JSON = "test.json"

    TRAIN_SPLIT_JSON = os.path.join("src", "data", "dataset_split", TRAIN_JSON)
    VAL_SPLIT_JSON = os.path.join("src", "data", "dataset_split", VAL_JSON)
    TEST_SPLIT_JSON = os.path.join("src", "data", "dataset_split", TEST_JSON)

    HPC_TRAIN_JSON = os.path.join(TMPDIR, TRAIN_JSON)
    HPC_VAL_JSON = os.path.join(TMPDIR, VAL_JSON)
    HPC_TEST_JSON = os.path.join(TMPDIR, TEST_JSON)

    # -------------------------
    # Data Shape Configs
    # -------------------------
    EXPECTED_SHAPE = (108, 134, 25, 25)
    UNET_COMPATIBLE_SHAPE = (144, 128)

    # -------------------------
    # DataLoader Configs
    # -------------------------
    BATCH_SIZE = 1
    NUM_WORKERS = 8

    # -------------------------
    # SSDDPM Configs
    # -------------------------
    SSDDPM_CONFIG = {
        # -------------------------
        # Epochs Configs
        # -------------------------
        "max_epochs": 250,  # MAX EPOCHS
        "log_every_n_steps": 8,
        # -------------------------
        # Noise Scheduler Configs
        # -------------------------
        "SCHEDULER_CONFIG": {
            "num_train_timesteps": 1000,  # T = 250
            "beta_start": 1e-7,  # β1 = 1e-7
            "beta_end": 2e-6,  # βT = 2e-6
            # "beta_start": 1e-4,  # β1 = 1e-4 (standard)
            # "beta_end": 0.02,  # βT = 0.02 (standard)
            # "beta_start": 1e-5,  # 10x your current start
            # "beta_end": 0.005,  # 2500x your current end, but 4x less than standard
            # "beta_start": 1e-6,  # 10x your current (not 1000x)
            # "beta_end": 1e-5,  # 5x your current (not 10,000x)
            "beta_schedule": "linear",  # Linear noise schedule
        },
        # -------------------------
        # Optimizer Configs
        # -------------------------
        "OPTIMIZER_CONFIG": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        # -------------------------
        # Model Configs
        # -------------------------
        "in_channels": 9,
        "out_channels": 9,
        # -------------------------
        # Loss Configs
        # -------------------------
        "lambda_adc": 0.25,
        "lambda_recon": 0.25,
        # -------------------------
        # Inference Configs
        # -------------------------
        "num_inference_steps": 250,
    }

    # -------------------------
    # ADC Configs
    # -------------------------
    ADC_CONFIG = {
        "adc_type": "avg",
        "num_dirs": 3,
        "n_bvals": 9,
    }

    # -------------------------
    # Logger Configs
    # -------------------------
    LOGGER_CONFIG = {
        "log_hyperparameters": True,  # Log hyperparameters
        "save_dir": "lightning_logs",
    }

    # -------------------------
    # Checkpoint Configs
    # -------------------------
    CHECKPOINT_CONFIG = {
        "save_dir": "checkpoints",
        "filename": "ssddpm-{epoch:02d}-val_loss={val_total_loss:.4f}",
        "monitor": "val_total_loss",
        "mode": "min",
        "save_top_k": 1,
        "every_n_epochs": SSDDPM_CONFIG["max_epochs"] // 4,
    }

    @classmethod
    def summary(cls):
        summary_dict = {
            "ORIGINAL_DATA_ROOT": cls.ORIGINAL_DATA_ROOT,
            "PT_DATA_ROOT": cls.PT_DATA_ROOT,
            "TMPDIR": cls.TMPDIR,
            "HPC_DATA_ROOT": cls.HPC_DATA_ROOT,
            "TRAIN_SPLIT_JSON": cls.TRAIN_SPLIT_JSON,
            "VAL_SPLIT_JSON": cls.VAL_SPLIT_JSON,
            "TEST_SPLIT_JSON": cls.TEST_SPLIT_JSON,
            "HPC_TRAIN_JSON": cls.HPC_TRAIN_JSON,
            "HPC_VAL_JSON": cls.HPC_VAL_JSON,
            "HPC_TEST_JSON": cls.HPC_TEST_JSON,
            "EXPECTED_SHAPE": cls.EXPECTED_SHAPE,
            "UNET_COMPATIBLE_SHAPE": cls.UNET_COMPATIBLE_SHAPE,
            "BATCH_SIZE": cls.BATCH_SIZE,
            "NUM_WORKERS": cls.NUM_WORKERS,
            "SSDDPM_CONFIG": cls.SSDDPM_CONFIG,
            "ADC_CONFIG": cls.ADC_CONFIG,
            "LOGGER_CONFIG": cls.LOGGER_CONFIG,
            "CHECKPOINT_CONFIG": cls.CHECKPOINT_CONFIG,
        }
        return pprint.pformat(summary_dict, indent=2)
