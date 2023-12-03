from pathlib import Path

from .dataset import BearingDataset


class CWRUDataset(BearingDataset):

    def __init__(self, signal_dir: Path, spectrogram_dir: Path) -> None:
        super().__init__(signal_dir, spectrogram_dir)

    def _download_signals(self, signal_dir: Path) -> None:
        pass

    def _make_spectrograms(self, spectrogram_dir: Path) -> None:
        pass
