import multiprocessing
from pathlib import Path
from typing import Literal

import lightning as pl
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import ImageClassificationDataset as Dataset
from hust_bearing.data.dataset import build_bearing_dataset
from hust_bearing.data.data import (
    BearingData,
    CWRU,
    HUST,
    list_bearing_data,
    split_bearing_data,
)


DataName = Literal["cwru", "hust"]


BEARING_DATA_CLASSES: dict[DataName, type[BearingData]] = {
    "cwru": CWRU,
    "hust": HUST,
}


class ImageClassificationDataModule(pl.LightningDataModule):
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
) -> ImageClassificationDataModule:
    bearing_data = BEARING_DATA_CLASSES[name]()

    paths = list_bearing_data(data_dir)
    labels = bearing_data.encode_labels(bearing_data.extract_labels(paths))
    loads = bearing_data.extract_loads(paths)

    (
        train_paths,
        test_paths,
        val_paths,
        train_labels,
        test_labels,
        val_labels,
    ) = split_bearing_data(paths, labels, loads, train_load, val_size)

    train_ds = build_bearing_dataset(train_paths, train_labels)
    test_ds = build_bearing_dataset(test_paths, test_labels)
    val_ds = build_bearing_dataset(val_paths, val_labels)
    return ImageClassificationDataModule(train_ds, test_ds, val_ds, batch_size)
