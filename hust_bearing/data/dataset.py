from pathlib import Path
from typing import Callable

import scipy
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from torch.utils.data import Dataset


class ImageClassificationDS(Dataset):
    def __init__(
        self,
        paths: npt.NDArray[np.object_],
        targets: npt.NDArray[np.int64],
        load_image: Callable[[Path], npt.NDArray[np.float32]],
        transform: Callable[[npt.NDArray], torch.Tensor],
    ) -> None:
        self._paths = paths
        self._labels = torch.from_numpy(targets)
        self._read_image = load_image
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = self._read_image(self._paths[idx])  # type: ignore
        image = self._transform(spectrogram)
        return image, self._labels[idx]


def bearing_dataset(
    paths: npt.NDArray[np.object_], targets: npt.NDArray[np.int64]
) -> ImageClassificationDS:
    return ImageClassificationDS(
        paths, targets, _load_spectrogram, _build_default_transform((64, 64))
    )


def _load_spectrogram(path: Path) -> npt.NDArray[np.float32]:
    data = scipy.io.loadmat(str(path))
    return data["spec"].astype(np.float32)


def _build_default_transform(
    image_size: tuple[int, int]
) -> Callable[[npt.NDArray[np.float32]], torch.Tensor]:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
        ]
    )
