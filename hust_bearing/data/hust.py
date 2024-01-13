import re
from pathlib import Path

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
    _classes = ["N", "B", "I", "O", "IB", "IO", "OB"]

    def target_from(self, path: Path) -> int:
        return self._classes.index(self._parse(path.parent.name).group(1))

    def load_from(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(3))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
