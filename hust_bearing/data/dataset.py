from pathlib import Path
from typing import Callable

import scipy
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from torch.utils.data import Dataset


class BearingDataset(Dataset):
    def __init__(
        self,
        paths: npt.NDArray[np.object_],
        targets: npt.NDArray[np.int64],
    ) -> None:
        self._paths = paths
        self._targets = torch.from_numpy(targets)
        self._transform = _build_transform((64, 64))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = _load_spectrogram(self._paths[idx])  # type: ignore
        image = self._transform(spectrogram)
        return image, self._targets[idx]


class DSANDataset(Dataset):
    def __init__(self, source_ds: BearingDataset, target_ds: BearingDataset) -> None:
        self._source_ds = source_ds
        self._target_ds = target_ds

    def __len__(self) -> int:
        return max(len(self._source_ds), len(self._target_ds))

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        source_idx = idx % len(self._source_ds)
        target_idx = idx % len(self._target_ds)
        return self._source_ds[source_idx], self._target_ds[target_idx]


def _load_spectrogram(path: Path) -> npt.NDArray[np.float32]:
    return scipy.io.loadmat(str(path))["data"].astype(np.float32)


def _build_transform(
    image_size: tuple[int, int]
) -> Callable[[npt.NDArray[np.float32]], torch.Tensor]:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
        ]
    )
