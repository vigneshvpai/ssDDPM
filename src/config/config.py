import os


class Config:
    # Paths and environment variables for data storage locations
    ORIGINAL_DATA_ROOT = "/home/vault/mfdp/mfdp118h/data"
    PT_DATA_ROOT = "/home/vault/mfdp/mfdp118h/pt_data"
    TMPDIR = os.environ.get("TMPDIR")
    HPC_DATA_ROOT = os.path.join(TMPDIR, "pt_data") if TMPDIR else None

    TRAIN_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "train.json")
    VAL_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "val.json")
    TEST_SPLIT_JSON = os.path.join("src", "data", "dataset_split", "test.json")

    EXPECTED_SHAPE = (108, 134, 25, 25)

    UNET_COMPATIBLE_SHAPE = (144, 128)

    # Batch size and number of workers for DataLoader
    BATCH_SIZE = 16
    NUM_WORKERS = 8

    # Scheduler config for SSDDPM
    SCHEDULER_CONFIG = {
        "num_train_timesteps": 250,  # T = 250
        "beta_start": 1e-7,  # β1 = 1e-7
        "beta_end": 2e-6,  # βT = 2e-6
        "beta_schedule": "linear",  # Linear noise schedule
    }

    # Optimizer config for SSDDPM
    OPTIMIZER_CONFIG = {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
