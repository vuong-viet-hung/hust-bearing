import re
from pathlib import Path


PathLike = Path | str


class HUSTParser:
    _dir_name_regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )

    def extract_label(self, path: PathLike) -> str:
        return self._parse(path).group(1)

    def extract_load(self, path: PathLike) -> str:
        return self._parse(path).group(3)

    def _parse(self, path: PathLike) -> re.Match[str]:
        dir_name = Path(path).parent.name
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid path: {path}")
        return match
