import re
from pathlib import Path
from typing import Callable

import lightning as pl
import numpy as np
import scipy
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class CWRU(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str = Path("data", "cqt", "cwru"),
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._paths = list(Path(data_dir).glob("**/*.mat"))
        self._batch_size = batch_size

    def prepare_data(self) -> None:
        encoder = LabelEncoder()
        labels = encoder.fit_transform([_extract_label(path) for path in self._paths])
        fit_paths, self._test_paths, fit_labels, self._test_labels = train_test_split(
            self._paths, labels, test_size=0.1, stratify=labels
        )
        (
            self._train_paths,
            self._val_paths,
            self._train_labels,
            self._val_labels,
        ) = train_test_split(
            fit_paths, fit_labels, test_size=0.1 / 0.9, stratify=fit_labels
        )
        self._transform = self._create_transform()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = Spectrograms(
                self._train_paths, self._train_labels, self._transform
            )
            self._val_ds = Spectrograms(
                self._val_paths, self._val_labels, self._transform
            )

        if stage == "test":
            self._test_ds = Spectrograms(
                self._test_paths, self._test_labels, self._transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self._batch_size, num_workers=8, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=self._batch_size, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=self._batch_size, num_workers=8)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _create_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        transforms = [torchvision.transforms.Resize((32, 32), antialias=None)]
        self._train_ds = Spectrograms(
            self._train_paths,
            self._train_labels,
            torchvision.transforms.Compose(transforms),
        )
        train_dl = self.train_dataloader()
        px_mean, px_std = self._compute_stats(train_dl)
        transforms.append(torchvision.transforms.Normalize(px_mean, px_std))
        return torchvision.transforms.Compose(transforms)

    def _compute_stats(self, data_loader: DataLoader) -> tuple[float, float]:
        px_sum = 0.0
        px_sum_sq = 0.0
        num_px = 0
        for image_batch, _ in data_loader:
            px_sum += image_batch.sum()
            px_sum_sq += (image_batch**2).sum()
            num_px += image_batch.numel()
        px_mean = px_sum / num_px
        px_var = px_sum_sq / num_px - px_mean**2
        px_std = px_var**0.5
        return px_mean, px_std


class Spectrograms(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._paths = paths
        self._labels = torch.from_numpy(labels)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = _load_spectrogram(self._paths[idx])
        image = self._transform(spectrogram.unsqueeze(dim=0))
        return image, self._labels[idx]


def _extract_label(path: Path) -> str:
    dir_name = path.parent.name
    match = re.fullmatch(
        r"""
        ([a-zA-Z]+)  # Fault
        (\d{3})?  # Fault size
        (@\d+)?  # Fault location
        _
        (\d+)  # Load
        """,
        dir_name,
        re.VERBOSE,
    )
    if match is None:
        raise ValueError(f"Invalid directory: {dir_name}")
    return match.group(1)


def _load_spectrogram(path: Path) -> torch.Tensor:
    data = scipy.io.loadmat(str(path))
    spectrogram = data["data"].astype(np.float32).T
    return torch.from_numpy(spectrogram)
