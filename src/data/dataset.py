from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch
import torchvision


Transform = Callable[[np.ndarray], torch.Tensor]


class BearingDataset(torchvision.datasets.ImageFolder, metaclass=ABCMeta):
    def __init__(self, signal_dir: Path, spectrogram_dir: Path) -> None:
        super().__init__(str(signal_dir), loader=scipy.io.loadmat)
        self._transform = self._make_transform()
        if not signal_dir.exists():
            self._download_signals(signal_dir)
        if not spectrogram_dir.exists():
            self._make_spectrograms(spectrogram_dir)

    def _make_transform(self) -> Transform:
        return torchvision.transforms.Compose(
            [torchvision.transforms.transforms.ToTensor(), torchvision.transforms.Normalize()]
        )

    @abstractmethod
    def _download_signals(self, signal_dir: Path) -> None:
        pass

    @abstractmethod
    def _make_spectrograms(self, spectrogram_dir: Path) -> None:
        pass
