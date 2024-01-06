from abc import ABCMeta, abstractmethod
from itertools import chain
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


class SpectrogramDM(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(self, data_dir: Path | str, batch_size: int) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            self.download()
        self.init_paths()
        self.init_labels()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = Spectrograms(self.train_paths, self.train_labels)
            self._val_ds = Spectrograms(self.val_paths, self.val_labels)

        elif stage == "validate":
            self._val_ds = Spectrograms(self.val_paths, self.val_labels)

        else:
            self._test_ds = Spectrograms(self.test_paths, self.test_labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_ds, self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, self.batch_size, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, self.batch_size, num_workers=8)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def extract_label(self, dir_name: str) -> str:
        pass

    @abstractmethod
    def init_paths(self) -> None:
        self.train_paths = []
        self.test_paths = []
        self.val_paths = []

    def init_labels(self) -> None:
        encoder_path = self.data_dir / "label_encoder.joblib"
        encoder = _load_encoder(encoder_path, self._get_train_labels())

        self.train_labels = encoder.transform(self._get_train_labels())
        self.test_labels = encoder.transform(self._get_test_labels())
        self.val_labels = encoder.transform(self._get_val_labels())

    def _get_train_labels(self) -> list[str]:
        return [self.extract_label(path.parent.name) for path in self.train_paths]

    def _get_test_labels(self) -> list[str]:
        return [self.extract_label(path.parent.name) for path in self.test_paths]

    def _get_val_labels(self) -> list[str]:
        return [self.extract_label(path.parent.name) for path in self.val_paths]


class MeasuredSpectrogramDM(SpectrogramDM, metaclass=ABCMeta):
    def __init__(self, train_load: str, data_dir: Path | str, batch_size: int) -> None:
        data_dir = Path(data_dir) / "measure"
        super().__init__(data_dir, batch_size)
        self._train_load = train_load

    def init_paths(self) -> None:
        dirs = self.data_dir.iterdir()

        loads = [self.extract_load(dir_.name) for dir_ in dirs]
        fit_dirs = [dir_ for dir_, load in zip(dirs, loads) if load == self._train_load]
        test_dirs = [
            dir_ for dir_, load in zip(dirs, loads) if load != self._train_load
        ]

        labels = [self.extract_label(dir_.name) for dir_ in fit_dirs]
        train_dirs, val_dirs = train_test_split(
            fit_dirs, test_size=0.2, stratify=labels
        )

        self.train_paths = _list_dirs(train_dirs)
        self.test_paths = _list_dirs(test_dirs)
        self.val_paths = _list_dirs(val_dirs)

    @abstractmethod
    def extract_load(self, dir_name: str) -> str:
        pass


class SimulatedSpectrogramDM(SpectrogramDM, metaclass=ABCMeta):
    def init_paths(self) -> None:
        test_dir = self.data_dir / "measure"
        fit_dir = self.data_dir / "simulate"

        test_dirs = list(test_dir.iterdir())
        fit_dirs = list(fit_dir.iterdir())
        fit_labels = [self.extract_label(dir_.name) for dir_ in fit_dirs]
        train_dirs, val_dirs = train_test_split(
            fit_dirs, test_size=0.2, stratify=fit_labels
        )

        self._train_paths = _list_dirs(train_dirs)
        self._test_paths = _list_dirs(test_dirs)
        self._val_paths = _list_dirs(val_dirs)


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


def _list_dirs(dirs: list[Path]) -> list[Path]:
    return list(chain.from_iterable(dir_.glob("*.mat") for dir_ in dirs))


def _load_encoder(encoder_path: Path, fit_labels: list[str]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(fit_labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _load_spectrogram(path: Path) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)
