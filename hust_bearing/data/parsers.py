import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt


class Parser(ABC):
    @abstractmethod
    def extract_label(self, path: Path) -> str:
        pass

    @abstractmethod
    def extract_load(self, path: Path) -> int:
        pass

    def extract_labels(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self.extract_label)(paths)

    def extract_loads(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self.extract_label)(paths)


class CWRUParser(Parser):
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

    def extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(4))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match


class HUSTParser(Parser):
    _dir_name_regex = re.compile(
        r"""
        ([A-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )

    def extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(3))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
