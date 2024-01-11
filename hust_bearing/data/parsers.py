import re
from pathlib import Path
from abc import ABC, abstractmethod


class Parser(ABC):
    @abstractmethod
    def extract_label(self, path: Path) -> str:
        pass

    @abstractmethod
    def extract_load(self, path: Path) -> str:
        pass


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
        return self._parse(path).group(1)

    def extract_load(self, path: Path) -> str:
        return self._parse(path).group(4)

    def _parse(self, path: Path) -> re.Match[str]:
        dir_name = path.parent.name
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
        return self._parse(path).group(1)

    def extract_load(self, path: Path) -> str:
        return self._parse(path).group(3)

    def _parse(self, path: Path) -> re.Match[str]:
        dir_name = path.parent.name
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError
        return match
