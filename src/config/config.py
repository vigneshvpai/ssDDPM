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
