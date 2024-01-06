from pathlib import Path

import numpy as np
import scipy
import torch
import torchvision
from torch.utils.data import Dataset


class Spectrograms(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: np.ndarray,
    ) -> None:
        self._paths = paths
        self._labels = torch.from_numpy(labels)
        self._transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64), antialias=None),
            ]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = _load_spectrogram(self._paths[idx])
        image = self._transform(spectrogram)
        return image, self._labels[idx]


def _load_spectrogram(path: Path) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)
