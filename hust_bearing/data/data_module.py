import multiprocessing
from pathlib import Path
from typing import Literal

import lightning as pl
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hust_bearing.data.dataset import ImageClassificationDS as Dataset
from hust_bearing.data.dataset import bearing_dataset
from hust_bearing.data.label_encoders import (
    LabelEncoder,
    CWRULabelEncoder,
    HUSTLabelEncoder,
)
from hust_bearing.data.path_parsers import (
    PathParser,
    CWRUPathParser,
    HUSTPathParser,
)


DataName = Literal["cwru", "hust"]


BEARING_DATA_CLASSES: dict[DataName, tuple[type[LabelEncoder], type[PathParser]]] = {
    "cwru": (CWRULabelEncoder, CWRUPathParser),
    "hust": (HUSTLabelEncoder, HUSTPathParser),
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
        return DataLoader(self._test_ds, self._batch_size, num_workers=self._num_worker)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, self._batch_size, num_workers=self._num_worker)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def bearing_data_module(
    name: DataName, data_dir: Path, batch_size: int, train_load: int
) -> ImageClassificationDM:
    paths, targets, loads = _from_bearing_data(name, data_dir)

    fit_paths = paths[loads == train_load]
    test_paths = paths[loads != train_load]
    fit_targets = targets[loads == train_load]
    test_targets = targets[loads != train_load]

    train_paths, val_paths, train_targets, val_targets = train_test_split(
        fit_paths, fit_targets, test_size=0.2, stratify=fit_targets
    )
    return ImageClassificationDM(
        bearing_dataset(train_paths, train_targets),
        bearing_dataset(test_paths, test_targets),
        bearing_dataset(val_paths, val_targets),
        batch_size,
    )


def _from_bearing_data(
    name: DataName, data_dir: Path
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    label_encoder_cls, path_parser_cls = BEARING_DATA_CLASSES[name]
    label_encoder = label_encoder_cls()
    path_parser = path_parser_cls()

    paths = np.array(list(data_dir.glob("*.mat")))
    labels = path_parser.extract_labels(paths)
    targets = label_encoder.encode(labels)
    loads = path_parser.extract_loads(paths)
    return paths, targets, loads
