import re
from pathlib import Path

from hust_bearing.data import SpectrogramDM


class HUSTParser(SpectrogramDM):
    _dir_name_regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )

    def _extract_label(self, path: Path | str) -> str:
        return self._parse(path).group(1)

    def _extract_load(self, path: Path | str) -> str:
        return self._parse(path).group(3)

    def _parse(self, path: Path | str) -> re.Match[str]:
        dir_name = path.parent.name
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid path: {path}")
        return match
