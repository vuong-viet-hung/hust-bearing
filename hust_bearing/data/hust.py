import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hust_bearing.data import BearingDataModule


class HUST(BearingDataModule):
    _dir_name_regex = re.compile(
        r"""
        ([A-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )
    _classes = np.array(["N", "B", "I", "O", "IB", "IO", "OB"])

    def __init__(
        self, data_dir: Path, batch_size: int, train_load: int, val_size: float
    ) -> None:
        super().__init__(data_dir, batch_size, train_load, val_size)

    def targets_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        labels = self.labels_from(paths)
        return np.argmax(np.expand_dims(self._classes, axis=1) == labels, axis=0)

    def labels_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.str_]:
        return np.vectorize(self._extract_label)(paths)

    def loads_from(self, paths: npt.NDArray[np.object_]) -> npt.NDArray[np.int64]:
        return np.vectorize(self._extract_load)(paths)

    def _extract_label(self, path: Path) -> str:
        return self._parse(path.parent.name).group(1)

    def _extract_load(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(3))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
