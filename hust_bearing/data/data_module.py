import multiprocessing
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
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

from hust_bearing.data import Parser, HUSTParser


class SpectrogramDM(pl.LightningDataModule, metaclass=ABCMeta):
    _parser_classes: dict[str, type[Parser]] = {
        "hust": HUSTParser,
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
        return DataLoader(
            self._test_ds, self._batch_size, num_workers=self._num_workers
        )

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
        paths = list(self._data_dir.glob("**/*.mat"))
        loads = self._extract_loads(paths)

        fit_paths = [
            path for path, load in zip(paths, loads) if load == self._train_load
        ]
        fit_labels = self._extract_labels(fit_paths)
        self._train_paths, self._val_paths = train_test_split(
            fit_paths, test_size=0.2, stratify=fit_labels
        )

        self._test_paths = [
            path for path, load in zip(paths, loads) if load != self._train_load
        ]

    def _init_labels(self) -> None:
        encoder_path = self._data_dir / ".encoder.joblib"
        encoder = _load_encoder(encoder_path, self._extract_labels(self._train_paths))

        train_labels = self._extract_labels(self._train_paths)
        test_labels = self._extract_labels(self._test_paths)
        val_labels = self._extract_labels(self._val_paths)

        self._train_labels = encoder.transform(train_labels)
        self._test_labels = encoder.transform(test_labels)
        self._val_labels = encoder.transform(val_labels)

    def _extract_loads(self, paths: Sequence[Path | str]) -> list[str]:
        return [self._parser.extract_load(path) for path in paths]

    def _extract_labels(self, paths: Sequence[Path | str]) -> list[str]:
        return [self._parser.extract_label(path) for path in paths]


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
    if Path(encoder_path).exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _load_spectrogram(path: Path | str) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)
