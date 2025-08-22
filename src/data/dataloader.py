import lightning as L
from torch.utils.data import DataLoader, Dataset


class DataLoader(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Optionally split dataset here for train/val/test
        self.data = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
