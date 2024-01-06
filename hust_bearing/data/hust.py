import re
from pathlib import Path

from hust_bearing.data.common import (
    MeasuredSpectrogramDM,
    SimulatedSpectrogramDM,
    Downloader,
    DirNameParser,
)


class HUSTDownloader(Downloader):
    def download(self, data_dir: Path | str) -> None:
        pass


class HUSTParser(DirNameParser):
    _regex = re.compile(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d)  # Bearing
        0
        (\d)  # Load
        """,
        re.VERBOSE,
    )

    def extract_label(self, dir_name: str) -> str:
        return self._parse(dir_name).group(1)

    def extract_load(self, dir_name: str) -> str:
        return self._parse(dir_name).group(3)

    def _parse(self, dir_name) -> re.Match[str]:
        match = self._regex.fullmatch(dir_name)
        if match is None:
            raise ValueError(f"Invalid directory name: {dir_name}")
        return match


class HUST(MeasuredSpectrogramDM):
    def __init__(
        self,
        train_load: str,
        data_dir: Path | str = Path("spectrograms", "hust"),
        batch_size: int = 32,
    ) -> None:
        downloader = HUSTDownloader()
        dir_name_parser = HUSTParser()
        super().__init__(data_dir, batch_size, downloader, dir_name_parser, train_load)


class HUSTSim(SimulatedSpectrogramDM):
    def __init__(
        self, data_dir: Path | str = Path("spectrograms", "hust"), batch_size: int = 32
    ) -> None:
        downloader = HUSTDownloader()
        dir_name_parser = HUSTParser()
        super().__init__(data_dir, batch_size, downloader, dir_name_parser)
