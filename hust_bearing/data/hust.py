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
        fit_dir = self._data_dir / "simulate"
        test_dir = self._data_dir / "measure"

        dirs: dict[str, list[Path]] = {}
        fit_dirs = list(fit_dir.iterdir())
        dirs["train"], dirs["val"] = train_test_split(
            fit_dirs,
            test_size=0.2,
            stratify=_labels_from_dirs(fit_dirs),
        )
        dirs["test"] = list(test_dir.iterdir())

        for stage in {"train", "test", "val"}:
            self._paths[stage] = dirs[stage]

    def _set_labels(self) -> None:
        encoder_path = self._data_dir / "label_encoder.joblib"
        encoder = _load_encoder(encoder_path, _labels_from_paths(self._paths["train"]))

        for stage in {"train", "test", "val"}:
            self._labels[stage] = encoder.transform(
                _labels_from_paths(self._paths[stage])
            )

    def _setup(self, stage: str) -> None:
        self._datasets[stage] = Spectrograms(self._paths[stage], self._labels[stage])


def _list_dirs(dirs: list[Path]) -> list[Path]:
    return list(chain.from_iterable(dir_.glob("*.mat") for dir_ in dirs))


def _load_encoder(encoder_path: Path, fit_labels: list[str]) -> LabelEncoder:
    if encoder_path.exists():
        return joblib.load(encoder_path)
    encoder = LabelEncoder()
    encoder.fit(fit_labels)
    joblib.dump(encoder, encoder_path)
    return encoder


def _labels_from_dirs(dirs: list[Path]) -> list[str]:
    return [_extract_label(dir_.name) for dir_ in dirs]


def _labels_from_paths(paths: list[Path]) -> list[str]:
    return [_extract_label(path.parent.name) for path in paths]


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
