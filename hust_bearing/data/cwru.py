import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hust_bearing.data import BearingDataModule


class CWRU(BearingDataModule):
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
    _classes = np.array(["Normal", "B", "IR", "OR"])

    def targets_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        labels = np.vectorize(self._extract_label)(paths)
        return np.argmax(np.expand_dims(self._classes, axis=1) == labels, axis=0)

    def loads_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self._extract_load)(paths)

    def _extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def _extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(4))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
