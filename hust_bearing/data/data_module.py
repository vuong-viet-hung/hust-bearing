from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import lightning as pl
import joblib
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import SpectrogramDS
from hust_bearing.data.parsers import Parser


T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Splits(Generic[T]):
    train: T
    test: T
    val: T

    def map(self, func: Callable[[Splits[T]], Splits[U]]) -> Splits[U]:
        return func(self)


class SpectrogramDM(pl.LightningDataModule):
    def __init__(
        self,
        parser: Parser,
        data_dir: Path,
        batch_size: int,
        train_load: str,
    ) -> None:
        super().__init__()
        self._parser = parser
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_load = train_load

        self._num_workers = multiprocessing.cpu_count()
        self._data_loaders: Splits[DataLoader] | None = None

    def prepare_data(self) -> None:
        paths = np.array(list(self._data_dir.glob("**/*.mat")))
        # fmt: off
        self._data_loaders = (
            self._split(paths)
            .map(self._to_datasets)
            .map(self._to_dataloaders)
        )
        # fmt: on

    def train_dataloader(self) -> DataLoader:
        if self._data_loaders is None:
            raise AttributeError
        return self._data_loaders.train

    def test_dataloader(self) -> DataLoader:
        if self._data_loaders is None:
            raise AttributeError
        return self._data_loaders.test

    def val_dataloader(self) -> DataLoader:
        if self._data_loaders is None:
            raise AttributeError
        return self._data_loaders.val

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _split(self, paths: npt.NDArray[np.object_]) -> Splits[npt.NDArray[np.object_]]:
        loads = self._extract_loads(paths)
        fit_paths = paths[loads == self._train_load]
        fit_labels = self._extract_labels(fit_paths)
        train_paths, val_paths = train_test_split(
            fit_paths, test_size=0.2, stratify=fit_labels
        )
        test_paths = paths[loads != self._train_load]
        return Splits(train_paths, test_paths, val_paths)

    def _to_labels(
        self, paths: Splits[npt.NDArray[np.object_]]
    ) -> Splits[npt.NDArray[np.str_]]:
        train_labels = self._extract_labels(paths.train)
        test_labels = self._extract_labels(paths.test)
        val_labels = self._extract_labels(paths.val)

        return Splits(train_labels, test_labels, val_labels)

    def _to_encoded_labels(
        self, labels: Splits[npt.NDArray[np.str_]]
    ) -> Splits[npt.NDArray[np.int64]]:
        encoder = LabelEncoder()
        encoder.fit(labels.train)

        train_labels = encoder.transform(labels.train)
        test_labels = encoder.transform(labels.test)
        val_labels = encoder.transform(labels.val)

        encoder_path = self._data_dir / ".encoder.joblib"
        joblib.dump(encoder, encoder_path)
        return Splits(train_labels, test_labels, val_labels)

    def _to_datasets(
        self, paths: Splits[npt.NDArray[np.object_]]
    ) -> Splits[SpectrogramDS]:
        # fmt: off
        labels = (
            paths
            .map(self._to_labels)
            .map(self._to_encoded_labels)
        )
        # fmt: on
        train_ds = SpectrogramDS(paths.train, labels.train)
        test_ds = SpectrogramDS(paths.test, labels.test)
        val_ds = SpectrogramDS(paths.val, labels.val)

        return Splits(train_ds, test_ds, val_ds)

    def _to_dataloaders(self, ds_splits: Splits) -> Splits[DataLoader]:
        train_dl = DataLoader(
            ds_splits.train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )
        test_dl = DataLoader(
            ds_splits.test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )
        val_dl = DataLoader(
            ds_splits.val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )
        return Splits(train_dl, test_dl, val_dl)

    def _extract_labels(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self._parser.extract_label)(paths)

    def _extract_loads(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self._parser.extract_load)(paths)
