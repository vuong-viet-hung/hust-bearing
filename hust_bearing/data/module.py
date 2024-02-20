import multiprocessing
from pathlib import Path

import lightning as pl
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import BearingDataset


class BearingDataModule(pl.LightningDataModule):
    _num_workers = multiprocessing.cpu_count()

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        load: int | None = None,
        num_samples: int | None = None,
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._load = load
        self._num_samples = num_samples
        empty_ds = BearingDataset([], [])
        self._train_ds = empty_ds
        self._test_ds = empty_ds
        self._val_ds = empty_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
