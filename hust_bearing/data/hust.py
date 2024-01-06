import re
from abc import ABC
from pathlib import Path

from hust_bearing.data.common import MeasuredSpectrogramDM, SimulatedSpectrogramDM


class HUSTMixin(ABC):
    def __init__(self, data_dir: Path | str, batch_size: int) -> None:
        super().__init__(data_dir, batch_size)
        self._dir_name_regex = re.compile(
            r"""
            ([a-zA-Z]+)  # Fault
            (\d)  # Bearing
            0
            (\d)  # Load
            """,
            re.VERBOSE,
        )

    def download(self) -> None:
        pass

    def extract_label(self, dir_name: str) -> str:
        return self.parse_dir_name(dir_name).group(1)

    def parse_dir_name(self, dir_name) -> re.Match[str]:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid directory name: {dir_name}")
        return match


class HUST(HUSTMixin, MeasuredSpectrogramDM):
    def __init__(
        self,
        train_load: str,
        data_dir: Path | str = Path("spectrograms", "hust"),
        batch_size: int = 32,
    ) -> None:
        super().__init__(train_load, data_dir, batch_size)

    def extract_load(self, dir_name: str) -> str:
        return self.parse_dir_name(dir_name).group(3)


class HUSTSim(HUSTMixin, SimulatedSpectrogramDM):
    def __init__(
        self, data_dir: Path | str = Path("spectrograms", "hust"), batch_size: int = 32
    ) -> None:
        super().__init__(data_dir, batch_size)

    def extract_label(self, dir_name: str) -> str:
        return self.parse_dir_name(dir_name).group(1)
