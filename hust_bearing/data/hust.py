import re

from hust_bearing.data.module import BearingDataModule


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

    def target_from(self, dir_name: str) -> int:
        return self._classes.index(self._parse(dir_name).group(1))

    def load_from(self, dir_name: str) -> int:
        return int(self._parse(dir_name).group(3))

    def _parse(self, dir_name: str) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
