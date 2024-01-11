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
        self._dl_splits: Splits[DataLoader] | None = None

    def prepare_data(self) -> None:
        # fmt: off
        self._dl_splits = (
            self._create_path_splits()
            .map(self._create_ds_splits)
            .map(self._create_dl_splits)
        )
        # fmt: on

    def train_dataloader(self) -> DataLoader:
        if self._dl_splits is None:
            raise AttributeError
        return self._dl_splits.train

    def test_dataloader(self) -> DataLoader:
        if self._dl_splits is None:
            raise AttributeError
        return self._dl_splits.test

    def val_dataloader(self) -> DataLoader:
        if self._dl_splits is None:
            raise AttributeError
        return self._dl_splits.val

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _create_path_splits(self) -> Splits[npt.NDArray[np.object_]]:
        paths = np.array(list(self._data_dir.glob("**/*.mat")))
        loads = self._extract_loads(paths)

        fit_paths = paths[loads == self._train_load]
        fit_labels = self._extract_labels(fit_paths)
        train_paths, val_paths = train_test_split(
            fit_paths, test_size=0.2, stratify=fit_labels
        )
        test_paths = paths[loads != self._train_load]
        return Splits(train_paths, test_paths, val_paths)

    def _create_label_splits(
        self, path_splits: Splits[npt.NDArray[np.object_]]
    ) -> Splits[npt.NDArray[np.str_]]:
        train_labels = self._extract_labels(path_splits.train)
        test_labels = self._extract_labels(path_splits.test)
        val_labels = self._extract_labels(path_splits.val)

        return Splits(train_labels, test_labels, val_labels)

    def _encode_label_splits(
        self, label_splits: Splits[npt.NDArray[np.str_]]
    ) -> Splits[npt.NDArray[np.int64]]:
        encoder_path = self._data_dir / ".encoder.joblib"
        encoder = _load_encoder(encoder_path, label_splits.train)

        train_labels = encoder.transform(label_splits.train)
        test_labels = encoder.transform(label_splits.test)
        val_labels = encoder.transform(label_splits.val)

        return Splits(train_labels, test_labels, val_labels)

    def _create_ds_splits(
        self, path_splits: Splits[npt.NDArray[np.object_]]
    ) -> Splits[SpectrogramDS]:
        # fmt: off
        label_splits = (
            path_splits
            .map(self._create_label_splits)
            .map(self._encode_label_splits)
        )
        # fmt: on
        train_ds = SpectrogramDS(path_splits.train, label_splits.train)
        test_ds = SpectrogramDS(path_splits.test, label_splits.test)
        val_ds = SpectrogramDS(path_splits.val, label_splits.val)

        return Splits(train_ds, test_ds, val_ds)

    def _create_dl_splits(self, ds_splits: Splits) -> Splits[DataLoader]:
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
        )
        val_dl = DataLoader(
            ds_splits.val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
        return Splits(train_dl, test_dl, val_dl)

    def _extract_labels(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self._parser.extract_label)(paths)

    def _extract_loads(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self._parser.extract_load)(paths)


def _load_encoder(encoder_path: Path, labels: npt.NDArray[np.str_]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, encoder_path)
    return encoder
