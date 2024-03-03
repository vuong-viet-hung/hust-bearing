from pathlib import Path
from typing import Callable

import scipy
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from torch.utils.data import Dataset


class BearingDataset(Dataset):
    def __init__(self, paths: list[Path], targets: list[int]) -> None:
        self._paths = paths
        self._targets = torch.tensor(targets)
        self._transform = _build_transform((32, 32))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = _load_spectrogram(self._paths[idx])
        image = self._transform(spectrogram)
        return image, self._targets[idx]


def _load_spectrogram(path: Path) -> npt.NDArray[np.float32]:
    return scipy.io.loadmat(str(path))["data"].astype(np.float32)


def _build_transform(
    image_size: tuple[int, int]
) -> Callable[[npt.NDArray[np.float32]], torch.Tensor]:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size, antialias=True),
        ]
    )
