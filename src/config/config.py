import os


class Config:
    # Paths and environment variables for data storage locations
    ORIGINAL_DATA_ROOT = "/home/vault/mfdp/mfdp118h/data"
    PT_DATA_ROOT = "/home/vault/mfdp/mfdp118h/pt_data"
    TMPDIR = os.environ.get("TMPDIR")
    HPC_DATA_ROOT = os.path.join(TMPDIR, "pt_data") if TMPDIR else None
