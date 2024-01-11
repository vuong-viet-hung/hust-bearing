import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split


class BearingData(ABC):
    @abstractmethod
    def extract_label(self, path: Path) -> str:
        pass

    @abstractmethod
    def extract_load(self, path: Path) -> int:
        pass

    @abstractmethod
    def encode_label(self, label: str) -> int:
        pass

    @abstractmethod
    def decode_label(self, label: int) -> str:
        pass

    def extract_labels(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self.extract_label)(paths)

    def extract_loads(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.extract_label)(paths)

    def encode_labels(self, labels: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.encode_label)(labels)

    def decode_labels(self, labels: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.encode_label)(labels)


class CWRU(BearingData):
    _dir_name_regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d{3})  # Size
        (?:@(\d+))?  # Location
        _
        (\d)  # Load
        """,
        re.VERBOSE,
    )
    _classes = ["Normal", "B", "IR", "OR"]

    def extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(4))

    def encode_label(self, label: str) -> int:
        return self._classes.index(label)

    def decode_label(self, label: int) -> str:
        return self._classes[label]

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match


class HUST(BearingData):
    _dir_name_regex = re.compile(
        r"""
        ([A-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )
    _classes = ["N", "B", "I", "O", "IB", "IO", "OB"]

    def extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(3))

    def encode_label(self, label: str) -> int:
        return self._classes.index(label)

    def decode_label(self, label: int) -> str:
        return self._classes[label]

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match


def list_bearing_data(data_dir: Path) -> npt.NDArray[np.object_]:
    return np.array(list(data_dir.glob("*.mat")))


def split_bearing_data(
    paths: npt.NDArray[np.object_],
    labels: npt.NDArray[np.int64],
    loads: npt.NDArray[np.int64],
    train_load: int,
    val_size: float,
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    fit_paths = paths[loads == train_load]
    test_paths = paths[loads != train_load]
    fit_labels = labels[loads == train_load]
    test_labels = labels[loads != train_load]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        fit_paths, fit_labels, test_size=val_size, stratify=fit_labels
    )
    return train_paths, test_paths, val_paths, train_labels, test_labels, val_labels
