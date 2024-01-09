from abc import ABC, abstractmethod
from pathlib import Path


PathLike = Path | str


class Parser(ABC):
    @abstractmethod
    def extract_label(self, path: PathLike) -> str:
        pass

    @abstractmethod
    def extract_load(self, path: PathLike) -> str:
        pass
