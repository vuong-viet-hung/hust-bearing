import multiprocessing
from pathlib import Path
from typing import Literal

import lightning as pl
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import ImageClassificationDS as Dataset
from hust_bearing.data.dataset import build_bearing_dataset
from hust_bearing.data.encoders import (
    Encoder,
    CWRUEncoder,
    HUSTEncoder,
)
from hust_bearing.data.parsers import (
    Parser,
    CWRUParser,
    HUSTParser,
)


DataName = Literal["cwru", "hust"]


BEARING_DATA_CLASSES: dict[DataName, tuple[type[Encoder], type[Parser]]] = {
    "cwru": (CWRUEncoder, CWRUParser),
    "hust": (HUSTEncoder, HUSTParser),
}


class ImageClassificationDM(pl.LightningDataModule):
    def __init__(
        self, train_ds: Dataset, test_ds: Dataset, val_ds: Dataset, batch_size: int
    ):
        super().__init__()
        self._train_ds = train_ds
        self._test_ds = test_ds
        self._val_ds = val_ds
        self._batch_size = batch_size
        self._num_worker = multiprocessing.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, self._batch_size, num_workers=self._num_worker, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds, self._batch_size, num_workers=self._num_worker, shuffle=False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, self._batch_size, num_workers=self._num_worker, shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def bearing_data_module(
    name: DataName, data_dir: Path, batch_size: int, train_load: int, val_size: float
) -> ImageClassificationDM:
    paths, labels, loads = _extract_from_bearing_data(name, data_dir)

    (
        train_paths,
        test_paths,
        val_paths,
        train_labels,
        test_labels,
        val_labels,
    ) = _split_bearing_data(paths, labels, loads, train_load, val_size)

    return ImageClassificationDM(
        build_bearing_dataset(train_paths, train_labels),
        build_bearing_dataset(test_paths, test_labels),
        build_bearing_dataset(val_paths, val_labels),
        batch_size,
    )


def _extract_from_bearing_data(
    name: DataName, data_dir: Path
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    encoder_cls, parser_cls = BEARING_DATA_CLASSES[name]
    encoder = encoder_cls()
    parser = parser_cls()

    paths = _list_bearing_data(data_dir)
    labels = encoder.encode_labels(parser.extract_labels(paths))
    loads = parser.extract_loads(paths)
    return paths, labels, loads


def _list_bearing_data(data_dir: Path) -> npt.NDArray[np.object_]:
    return np.array(list(data_dir.glob("*.mat")))


def _split_bearing_data(
    paths: npt.NDArray[np.object_],
    labels: npt.NDArray[np.int64],
    loads: npt.NDArray[np.int64],
    train_load: int,
    val_size: float,
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    fit_paths = paths[loads == train_load]
    test_paths = paths[loads != train_load]
    fit_labels = labels[loads == train_load]
    test_labels = labels[loads != train_load]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        fit_paths, fit_labels, test_size=val_size, stratify=fit_labels
    )
    return train_paths, test_paths, val_paths, train_labels, test_labels, val_labels
