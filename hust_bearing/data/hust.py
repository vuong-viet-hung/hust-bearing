import re
from itertools import chain
from pathlib import Path

import lightning as pl
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from hust_bearing.data.common import Spectrograms


class HUSTSim(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str = Path("spectrograms", "hust"),
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._data_dir = Path(data_dir)
        self._batch_size = batch_size
        self._paths: dict[str, list[Path]] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._datasets: dict[str, Spectrograms] = {}

    def prepare_data(self) -> None:
        self._set_paths()
        self._set_labels()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._setup("train")
            self._setup("val")

        elif stage == "validate":
            self._setup("val")

        else:
            self._setup("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["train"],
            batch_size=self._batch_size,
            num_workers=8,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["test"], batch_size=self._batch_size, num_workers=8
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["val"], batch_size=self._batch_size, num_workers=8
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _set_paths(self) -> None:
        subdirs = _list_data_dir(self._data_dir, val_size=0.2)
        for split in {"train", "test", "val"}:
            self._paths[split] = _list_subdirs(subdirs[split])

    def _set_labels(self) -> None:
        encoder_path = self._data_dir / "label_encoder.joblib"
        encoder = _load_encoder(encoder_path, _labels_from_paths(self._paths["train"]))
        for split in {"train", "test", "val"}:
            self._labels[split] = encoder.transform(
                _labels_from_paths(self._paths[split])
            )

    def _setup(self, split: str) -> None:
        self._datasets[split] = Spectrograms(self._paths[split], self._labels[split])


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


def _list_data_dir(data_dir: Path, val_size: float) -> dict[str, list[Path]]:
    fit_dir = data_dir / "simulate"
    test_dir = data_dir / "measure"
    subdirs: dict[str, list[Path]] = {}
    fit_subdirs = list(fit_dir.iterdir())
    subdirs["train"], subdirs["val"] = train_test_split(
        fit_subdirs,
        test_size=val_size,
        stratify=_labels_from_subdirs(fit_subdirs),
    )
    subdirs["test"] = list(test_dir.iterdir())
    return subdirs


def _load_encoder(encoder_path: Path, fit_labels: list[str]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(fit_labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _list_subdirs(subdirs: list[Path]) -> list[Path]:
    return list(chain.from_iterable(subdir.glob("*.mat") for subdir in subdirs))


def _labels_from_subdirs(subdirs: list[Path]) -> list[str]:
    return [_extract_label(subdir.name) for subdir in subdirs]


def _labels_from_paths(paths: list[Path]) -> list[str]:
    return [_extract_label(path.parent.name) for path in paths]
