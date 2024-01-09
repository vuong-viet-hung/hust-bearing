import functools
import multiprocessing
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import lightning as pl
import joblib
import numpy as np
import scipy
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from hust_bearing.data import Parser, HUSTParser


PathLike = Path | str
T = TypeVar("T")


class Splits(NamedTuple, Generic[T]):
    train: T
    test: T
    val: T


class SpectrogramDM(pl.LightningDataModule):
    _parser_classes: dict[str, type[Parser]] = {
        "hust": HUSTParser,
    }

    def __init__(
        self,
        name: str,
        train_load: str,
        data_dir: PathLike,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._train_load = train_load
        self._data_dir = Path(data_dir)
        self._parser = self._parser_classes[name]()
        self._ds_splits: Splits[SpectrogramDS] | None = None
        self.SpectrogramDL = functools.partial(
            DataLoader, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
        )

    def prepare_data(self) -> None:
        path_splits = self._get_path_splits()
        label_splits = self._get_label_splits(path_splits)
        self._set_ds_splits(path_splits, label_splits)

    def train_dataloader(self) -> DataLoader:
        if self._ds_splits is None:
            raise ValueError("Dataset hasn't been created")
        return self.SpectrogramDL(self._ds_splits.train)

    def test_dataloader(self) -> DataLoader:
        if self._ds_splits is None:
            raise ValueError("Dataset hasn't been created")
        return self.SpectrogramDL(self._ds_splits.test)

    def val_dataloader(self) -> DataLoader:
        if self._ds_splits is None:
            raise ValueError("Dataset hasn't been created")
        return self.SpectrogramDL(self._ds_splits.val)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _get_path_splits(self) -> Splits[np.ndarray]:
        paths = np.array(list(self._data_dir.glob("**/*.mat")))
        loads = self._extract_loads(paths)

        fit_paths = paths[loads == self._train_load]
        fit_labels = self._extract_labels(fit_paths)
        train_paths, val_paths = train_test_split(
            fit_paths, test_size=0.2, stratify=fit_labels
        )
        test_paths = paths[loads != self._train_load]
        return Splits(train_paths, test_paths, val_paths)

    def _get_label_splits(self, path_splits: Splits[np.ndarray]) -> Splits[np.ndarray]:
        train_labels = self._extract_labels(path_splits.train)
        test_labels = self._extract_labels(path_splits.test)
        val_labels = self._extract_labels(path_splits.val)

        encoder_path = self._data_dir / ".encoder.joblib"
        encoder = _load_encoder(encoder_path, train_labels)
        return Splits(
            encoder.transform(train_labels),
            encoder.transform(test_labels),
            encoder.transform(val_labels),
        )

    def _set_ds_splits(
        self, path_splits: Splits[np.ndarray], label_splits: Splits[np.ndarray]
    ) -> None:
        self._ds_splits = Splits(
            SpectrogramDS(path_splits.train, label_splits.train),
            SpectrogramDS(path_splits.test, label_splits.test),
            SpectrogramDS(path_splits.val, label_splits.val),
        )

    def _extract_loads(self, paths: np.ndarray) -> np.ndarray:
        return np.vectorize(self._parser.extract_load)(paths)

    def _extract_labels(self, paths: np.ndarray) -> np.ndarray:
        return np.vectorize(self._parser.extract_label)(paths)


class SpectrogramDS(Dataset):
    def __init__(
        self,
        paths: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._paths = paths
        self._labels = torch.from_numpy(labels)
        self._transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64), antialias=None),
            ]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = _load_spectrogram(self._paths[idx])  # type: ignore
        image = self._transform(spectrogram)
        return image, self._labels[idx]


def _load_encoder(encoder_path: PathLike, labels: np.ndarray) -> LabelEncoder:
    if Path(encoder_path).exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _load_spectrogram(path: PathLike) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)
