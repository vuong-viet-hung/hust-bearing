import re
from pathlib import Path

from hust_bearing.data.common import SimulatedSpectrogramDM


class HUSTSim(SimulatedSpectrogramDM):
    def __init__(
        self, data_dir: Path | str = Path("spectrograms", "hust"), batch_size: int = 32
    ) -> None:
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

    def extract_label(self, dir_name: str) -> str:
        match = self._dir_name_regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid directory name: {dir_name}")
        return match.group(1)
