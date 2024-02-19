import multiprocessing
from abc import ABCMeta, abstractmethod
from pathlib import Path

import lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import BearingDataset


class BearingDataModule(pl.LightningDataModule, metaclass=ABCMeta):
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

    def setup(self, stage: str) -> None:
        paths = list(self._data_dir.glob("**/*.mat"))

        if self._load is not None:
            paths = [
                path for path in paths if self.load_from(path.parent.name) == self._load
            ]

        targets = [self.target_from(path.parent.name) for path in paths]

        if self._num_samples is not None:
            _, paths, _, targets = train_test_split(
                paths, targets, test_size=self._num_samples, stratify=targets
            )

        if stage in {"fit", "validate"}:
            train_paths, val_paths, train_targets, val_targets = train_test_split(
                paths, targets, test_size=0.3, stratify=targets
            )
            self._train_ds = BearingDataset(train_paths, train_targets)
            self._val_ds = BearingDataset(val_paths, val_targets)

        elif stage in {"test", "predict"}:
            self._test_ds = BearingDataset(paths, targets)

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

    @abstractmethod
    def target_from(self, dir_name: str) -> int:
        pass

    @abstractmethod
    def load_from(self, dir_name: str) -> int:
        pass