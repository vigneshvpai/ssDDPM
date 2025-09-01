import os


class Config:
    # -------------------------
    # Path and Environment Configs
    # -------------------------
    ORIGINAL_DATA_ROOT = "/home/vault/mfdp/mfdp118h/data"
    PT_DATA_ROOT = "/home/vault/mfdp/mfdp118h/pt_data"
    TMPDIR = os.environ.get("TMPDIR")
    HPC_DATA_ROOT = os.path.join(TMPDIR, "pt_data") if TMPDIR else None

    TRAIN_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "train.json")
    VAL_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "val.json")
    TEST_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "test.json")
    TEST_NIFTI_JSON_PATH = os.path.join(
        "src", "data", "dataset_split", "test_nifti.json"
    )

    # -------------------------
    # Data Shape Configs
    # -------------------------
    EXPECTED_SHAPE = (108, 134, 25, 25)
    UNET_COMPATIBLE_SHAPE = (144, 128)

    # -------------------------
    # DataLoader Configs
    # -------------------------
    BATCH_SIZE = 2
    NUM_WORKERS = 8

    # -------------------------
    # SSDDPM Configs
    # -------------------------
    SSDDPM_CONFIG = {
        # -------------------------
        # Scheduler Configs
        # -------------------------
        "SCHEDULER_CONFIG": {
            "num_train_timesteps": 250,  # T = 250
            "beta_start": 1e-7,  # β1 = 1e-7
            "beta_end": 2e-6,  # βT = 2e-6
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
        "in_channels": 625,
        "out_channels": 625,
        "lambda_reg": 1,
        "num_inference_steps": 250,
        "max_epochs": 10,
    }

    # -------------------------
    # ADC Configs
    # -------------------------
    ADC_CONFIG = {
        "adc_type": "avg",  # avg or dir
        "num_dirs": 3,  # only used if adc_type == "dir"
        "n_bvals": 25,
    }
