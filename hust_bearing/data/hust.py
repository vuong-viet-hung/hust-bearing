import re
from pathlib import Path

from hust_bearing.data.common import SpectrogramDM


class HUST(SpectrogramDM):
    _regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )
    _url = "https://drive.google.com/file/d/1AjDHJgnxnjrhW5biEirtK3s6IrFL7rFV/view?usp=sharing"
    _download_file_path = Path("spectrograms.tar.gz")

    def __init__(
        self,
        train_load: str,
        data_dir: Path | str = Path(),
        batch_size: int = 32,
    ) -> None:
        super().__init__(train_load, data_dir, batch_size)

    def extract_label(self, dir_name: str) -> str:
        return self._parse(dir_name).group(1)

    def extract_load(self, dir_name: str) -> str:
        return self._parse(dir_name).group(3)

    def _parse(self, dir_name) -> re.Match[str]:
        match = self._regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid directory name: {dir_name}")
        return match
