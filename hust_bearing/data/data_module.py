import multiprocessing
from abc import ABCMeta, abstractmethod
from pathlib import Path

import lightning as pl
import numpy as np
import numpy.typing as npt
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import BearingDataset


class BearingDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self, data_dir: Path, batch_size: int, train_load: int, val_size: float
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_load = train_load
        self._val_size = val_size
        self._num_workers = multiprocessing.cpu_count()

        self._train_ds: BearingDataset | None = None
        self._test_ds: BearingDataset | None = None
        self._val_ds: BearingDataset | None = None

    def setup(self, stage: str) -> None:
        paths = np.array(list(self._data_dir.glob("*.mat")))
        targets = self.targets_from(paths)
        loads = self.loads_from(paths)

        fit_paths = paths[loads == self._train_load]
        test_paths = paths[loads != self._train_load]
        fit_targets = targets[loads == self._train_load]
        test_targets = targets[loads != self._train_load]

        train_paths, val_paths, train_targets, val_targets = train_test_split(
            fit_paths, fit_targets, test_size=0.2, stratify=fit_targets
        )

        if stage == "fit":
            self._train_ds = BearingDataset(train_paths, train_targets)
            self._val_ds = BearingDataset(val_paths, val_targets)

        if stage == "validate":
            self._val_ds = BearingDataset(val_paths, val_targets)

        if stage in ("test", "predict"):
            self._test_ds = BearingDataset(test_paths, test_targets)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._train_ds is None:
            raise AttributeError
        return DataLoader(
            self._train_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test_ds is None:
            raise AttributeError
        return DataLoader(
            self._test_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val_ds is None:
            raise AttributeError
        return DataLoader(
            self._val_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    @abstractmethod
    def targets_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        pass

    @abstractmethod
    def loads_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        pass
