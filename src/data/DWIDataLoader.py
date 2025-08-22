import lightning as L
from torch.utils.data import DataLoader
from src.data.DWIDataset import DWIDataset
from src.config.config import Config
from src.data.Preprocess import Preprocess


class DWIDataLoader(L.LightningDataModule):
    def __init__(
        self,
        train_json=Config.TRAIN_SPLIT_JSON,
        val_json=Config.VAL_SPLIT_JSON,
        test_json=Config.TEST_SPLIT_JSON,
        batch_size: int = Config.BATCH_SIZE,
        num_workers: int = Config.NUM_WORKERS,
        transform=None,
    ):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.preprocess_fn = Preprocess().preprocess

    def setup(self, stage=None):
        self.train_dataset = DWIDataset(
            split_json_path=self.train_json,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
        )
        self.val_dataset = DWIDataset(
            split_json_path=self.val_json,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
        )
        self.test_dataset = DWIDataset(
            split_json_path=self.test_json,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
