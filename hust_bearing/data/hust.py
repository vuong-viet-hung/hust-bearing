# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes

import re
from itertools import chain
from pathlib import Path

import lightning as pl
import joblib
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

    def prepare_data(self) -> None:
        self._init_paths()
        self._init_labels()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = Spectrograms(self._train_paths, self._train_labels)
            self._val_ds = Spectrograms(self._test_paths, self._test_labels)

        elif stage == "validate":
            self._val_ds = Spectrograms(self._val_paths, self._val_labels)

        else:
            self._test_ds = Spectrograms(self._test_paths, self._test_labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            num_workers=8,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=self._batch_size, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=self._batch_size, num_workers=8)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _init_paths(self) -> None:
        fit_dir = self._data_dir / "simulate"
        test_dir = self._data_dir / "measure"

        fit_dirs = list(fit_dir.iterdir())
        test_dirs = list(test_dir.iterdir())
        train_dirs, val_dirs = train_test_split(
            fit_dirs,
            test_size=0.2,
            stratify=_labels_from_dirs(fit_dirs),
        )

        self._train_paths = _list_dirs(train_dirs)
        self._test_paths = _list_dirs(test_dirs)
        self._val_paths = _list_dirs(val_dirs)

    def _init_labels(self) -> None:
        encoder_path = self._data_dir / "label_encoder.joblib"
        encoder = _load_encoder(encoder_path, _labels_from_paths(self._train_paths))
        self._train_labels = encoder.transform(_labels_from_paths(self._train_paths))
        self._test_labels = encoder.transform(_labels_from_paths(self._test_paths))
        self._val_labels = encoder.transform(_labels_from_paths(self._val_paths))


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


def _load_encoder(encoder_path: Path, fit_labels: list[str]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(fit_labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _list_dirs(dirs: list[Path]) -> list[Path]:
    return list(chain.from_iterable(dir_.glob("*.mat") for dir_ in dirs))


def _labels_from_dirs(dirs: list[Path]) -> list[str]:
    return [_extract_label(dir_.name) for dir_ in dirs]


def _labels_from_paths(paths: list[Path]) -> list[str]:
    return [_extract_label(path.parent.name) for path in paths]
