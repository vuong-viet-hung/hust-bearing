import re
from pathlib import Path

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
    _classes = ["Normal", "B", "IR", "OR"]

    def target_from(self, path: Path) -> int:
        return self._classes.index(self._parse(path.parent.name).group(1))

    def load_from(self, path: Path) -> int:
        return int(self._parse(path.parent.name).group(4))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
