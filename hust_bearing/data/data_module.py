from abc import ABCMeta, abstractmethod
import multiprocessing
from pathlib import Path

import lightning as pl
import joblib
import numpy as np
import scipy
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from hust_bearing.data import HUST


class SpectrogramDM(pl.LightningDataModule, metaclass=ABCMeta):
    _parser_classes = {
        "hust": HUST,
    }

    def __init__(
        self,
        name: str,
        train_load: str,
        data_dir: Path | str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._parser = self._parser_classes[name]()

        self._train_load = train_load
        self._data_dir = Path(data_dir)
        self._batch_size = batch_size

        self._num_workers = multiprocessing.cpu_count()

    def prepare_data(self) -> None:
        self._init_paths()
        self._init_labels()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = Spectrograms(self._train_paths, self._train_labels)
            self._val_ds = Spectrograms(self._val_paths, self._val_labels)

        elif stage == "validate":
            self._val_ds = Spectrograms(self._val_paths, self._val_labels)

        else:
            self._test_ds = Spectrograms(self._test_paths, self._test_labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, self._batch_size, num_workers=self._num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @abstractmethod
    def _extract_label(self, dir_name: str) -> str:
        pass

    @abstractmethod
    def _extract_load(self, dir_name: str) -> str:
        pass

    def _init_paths(self) -> None:
        fit_paths = self._get_fit_paths()
        fit_labels = [self._extract_label(path.parent.name) for path in fit_paths]
        self._train_paths, self._val_paths = train_test_split(
            fit_paths, test_size=0.2, stratify=fit_labels
        )
        self._test_paths = self._get_test_paths()

    def _init_labels(self) -> None:
        encoder_path = self._data_dir / ".encoder.joblib"
        encoder = _load_encoder(encoder_path, self._get_train_labels())

        self._train_labels = encoder.transform(self._get_train_labels())
        self._test_labels = encoder.transform(self._get_test_labels())
        self._val_labels = encoder.transform(self._get_val_labels())

    def _get_fit_paths(self) -> list[Path]:
        return [
            path
            for path in self._data_dir.glob("**/*.mat")
            if self._parser._extract_load(path) == self._train_load
        ]

    def _get_test_paths(self) -> list[Path]:
        return [
            path
            for path in self._data_dir.glob("**/*.mat")
            if self._parser._extract_load(path) != self._train_load
        ]

    def _get_train_labels(self) -> list[str]:
        return [self._parser._extract_label(path) for path in self._train_paths]

    def _get_test_labels(self) -> list[str]:
        return [self._parser._extract_label(path) for path in self._test_paths]

    def _get_val_labels(self) -> list[str]:
        return [self._parser._extract_label(path) for path in self._val_paths]


class Spectrograms(Dataset):
    def __init__(
        self,
        paths: list[Path],
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
        spectrogram = _load_spectrogram(self._paths[idx])
        image = self._transform(spectrogram)
        return image, self._labels[idx]


def _load_encoder(encoder_path: Path | str, labels: list[str]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _load_spectrogram(path: Path | str) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)