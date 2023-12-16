import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Protocol, TypeGuard, TypeVar
from typing_extensions import Self

import numpy as np
import torchvision
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch


class SizedDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def _is_sized(dataset: Dataset) -> TypeGuard[SizedDataset]:
    return hasattr(dataset, "__len__")


class SegmentSTFTs(Dataset):
    def __init__(
        self,
        data_file: Path,
        target: int,
        seg_length: int,
        win_length: int,
        hop_length: int,
        loader: Callable[[Path], np.ndarray],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._data_file = data_file
        self._target = torch.tensor(target)
        self._seg_length = seg_length
        self._win_length = win_length
        self._hop_length = hop_length
        self._loader = loader
        self._transform = transform
        signal = loader(data_file)
        self._num_segments = len(signal) // self._seg_length

    def __len__(self) -> int:
        return self._num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self._loader(self._data_file)
        segment = signal[idx * self._seg_length : (idx + 1) * self._seg_length]
        stft = torch.stft(
            torch.tensor(segment),
            self._win_length,
            self._hop_length,
            return_complex=True,
        )
        amplitude = stft.abs()
        db = 20 * amplitude.log10()
        image = self._transform(db.unsqueeze(dim=0))
        return image, self._target


class TransformDataset(Dataset):
    def __init__(
        self, dataset: Dataset, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        if _is_sized(self._dataset):
            return len(self._dataset)
        raise AttributeError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[idx]
        image = self._transform(image)
        return image, label


class Pipeline(ABC):
    def __init__(self) -> None:
        self.data_loaders: dict[str, DataLoader] = {}
        self.label_encoder = LabelEncoder()
        self._data_dir: Path | None = None
        self._dataset: Dataset | None = None
        self._subsets: dict[str, Dataset] = {}
        self._batch_size: int = 1
        self._num_workers: int = 1
        self._pixel_min: float = float("inf")
        self._pixel_max: float = float("-inf")
        self._pixel_mean: float = 0.0
        self._pixel_std: float = 0.0

    def download(self, data_dir: Path) -> Self:
        if not data_dir.exists():
            logging.info("Downloading data to '%s'...", data_dir)
            self._download(data_dir)
        logging.info("Data downloaded at '%s'", data_dir)
        self._data_dir = data_dir
        return self

    def build_dataset(
        self,
        image_size: tuple[int, int],
        seg_length: int,
        win_length: int,
        hop_length: int,
    ) -> Self:
        if self._data_dir is None:
            raise ValueError("Data isn't downloaded.")
        get_segment_stfts = functools.partial(
            SegmentSTFTs,
            seg_length=seg_length,
            win_length=win_length,
            hop_length=hop_length,
            loader=self._read_signal,
            transform=torchvision.transforms.Resize(image_size, antialias=None),
        )
        data_files = self._list_data_files(self._data_dir)
        data_files.sort()  # For deterministic dataset splitting
        labels = [self._read_label(file) for file in data_files]
        targets = self.label_encoder.fit_transform(labels)
        self._dataset = ConcatDataset(
            [
                get_segment_stfts(file, target)
                for file, target in zip(data_files, targets)
            ]
        )
        return self

    def split_dataset(self, fractions: tuple[float, float, float]) -> Self:
        if self._dataset is None:
            raise ValueError("Dataset isn't built.")
        (
            self._subsets["train"],
            self._subsets["valid"],
            self._subsets["test"],
        ) = random_split(self._dataset, fractions)
        return self

    def build_data_loaders(self, batch_size: int, num_workers: int) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self._subsets.keys()):
            raise ValueError("Dataset isn't built or split.")
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._build_data_loader("train")
        self._build_data_loader("valid")
        self._build_data_loader("test")
        return self

    def min_max_scale(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders aren't built.")
        logging.info("Min-max scaling data...")
        self._min_max_scale("train")
        self._min_max_scale("valid")
        self._min_max_scale("test")
        logging.info("Data min-max scaled")
        return self

    def normalize(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders aren't built.")
        logging.info("Normalizing data...")
        self._normalize("train")
        self._normalize("valid")
        self._normalize("test")
        logging.info("Data normalized")
        return self

    def _build_data_loader(self, subset: str) -> None:
        self.data_loaders[subset] = DataLoader(
            self._subsets[subset],
            self._batch_size,
            shuffle=(subset == "train"),
            num_workers=self._num_workers,
        )

    def _min_max_scale(self, subset: str) -> None:
        if self._pixel_min == float("inf"):
            self._compute_stats()
        loc = (self._pixel_max + self._pixel_min) / 2
        scale = (self._pixel_max - self._pixel_min) / 2
        scaler = torchvision.transforms.Normalize(loc, scale)
        self._subsets[subset] = TransformDataset(self._subsets[subset], scaler)
        self._build_data_loader(subset)

    def _normalize(self, subset: str) -> None:
        if self._pixel_min == float("inf"):
            self._compute_stats()
        normalizer = torchvision.transforms.Normalize(self._pixel_mean, self._pixel_std)
        self._subsets[subset] = TransformDataset(self._subsets[subset], normalizer)
        self._build_data_loader(subset)

    def _compute_stats(self) -> None:
        pixel_sum = 0.0
        num_pixels = 0
        data_loader = self.data_loaders["train"]

        for image_batch, _ in data_loader:
            self._pixel_min = min(self._pixel_min, image_batch.min().item())
            self._pixel_max = max(self._pixel_max, image_batch.max().item())
            pixel_sum += image_batch.sum().item()
            num_pixels += image_batch.numel().item()

        self._pixel_min = pixel_sum / num_pixels

        pixel_ssd = sum(
            ((image_batch - self._pixel_mean) ** 2).sum().item()
            for image_batch, _ in data_loader
        )

        self._pixel_std = (pixel_ssd / num_pixels) ** 0.5

    @abstractmethod
    def _download(self, data_dir: Path) -> None:
        pass

    @abstractmethod
    def _list_data_files(self, data_dir: Path) -> list[Path]:
        pass

    @abstractmethod
    def _read_label(self, data_file: Path) -> str:
        pass

    @abstractmethod
    def _read_signal(self, data_file: Path) -> np.ndarray:
        pass


P = TypeVar("P", bound=type[Pipeline])
_pipeline_registry: dict[str, type[Pipeline]] = {}


def register_pipeline(name: str) -> Callable[[P], P]:
    def decorator(pipeline_cls: P) -> P:
        _pipeline_registry[name] = pipeline_cls
        return pipeline_cls

    return decorator


def build_pipeline(name: str) -> Pipeline:
    if name not in _pipeline_registry:
        raise ValueError(f"Unregistered dataset: {name}")
    pipeline_cls = _pipeline_registry[name]
    return pipeline_cls()
