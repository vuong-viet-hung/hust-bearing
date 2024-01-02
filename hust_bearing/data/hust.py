import re
from itertools import chain
from pathlib import Path

import lightning as pl
import numpy as np
import scipy
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class HUSTSim(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str = Path("spectrograms", "hust"),
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._data_dir = Path(data_dir)
        self._batch_size = batch_size

    def prepare_data(self) -> None:
        fit_dir = self._data_dir / "simulate"
        test_dir = self._data_dir / "measure"

        fit_dirs = list(fit_dir.iterdir())
        train_dirs, val_dirs = train_test_split(
            fit_dirs,
            test_size=0.2,
            stratify=_labels_from_dirs(fit_dirs),
        )
        encoder = LabelEncoder()

        self._train_paths = _list_dirs(train_dirs)
        self._test_paths = list(test_dir.glob("**/*.mat"))
        self._val_paths = _list_dirs(val_dirs)

        self._train_labels = encoder.fit_transform(
            _labels_from_paths(self._train_paths)
        )
        self._test_labels = encoder.transform(_labels_from_paths(self._test_paths))
        self._val_labels = encoder.transform(_labels_from_paths(self._val_paths))

        print("Classes:", encoder.classes_)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = Spectrograms(self._train_paths, self._train_labels)
            self._val_ds = Spectrograms(self._val_paths, self._val_labels)

        if stage == "test":
            self._test_ds = Spectrograms(self._test_paths, self._test_labels)

        if stage == "predict":
            self._predict_ds = Spectrograms(self._test_paths, self._test_labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self._batch_size, num_workers=8, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=self._batch_size, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=self._batch_size, num_workers=8)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._predict_ds, batch_size=self._batch_size, num_workers=8)


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


def _extract_label(name: str) -> str:
    match = re.fullmatch(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        name,
        re.VERBOSE,
    )
    if match is None:
        raise ValueError(f"Invalid name: {name}")
    return match.group(1)


def _list_dirs(dirs: list[Path]) -> list[Path]:
    return list(chain.from_iterable(dir_.glob("*.mat") for dir_ in dirs))


def _labels_from_dirs(dirs: list[Path]) -> list[str]:
    return [_extract_label(dir_.name) for dir_ in dirs]


def _labels_from_paths(paths: list[Path]) -> list[str]:
    return [_extract_label(path.parent.name) for path in paths]


def _load_spectrogram(path: Path) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    spectrogram = data["spec"].astype(np.float32)
    return spectrogram
