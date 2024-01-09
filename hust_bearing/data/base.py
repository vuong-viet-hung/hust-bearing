from abc import ABC, abstractmethod
from pathlib import Path


class Parser(ABC):
    @abstractmethod
    def extract_label(self, path: Path | str) -> str:
        pass

    @abstractmethod
    def extract_load(self, path: Path | str) -> str:
        pass
