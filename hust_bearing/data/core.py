import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, Protocol, TypeGuard, TypeVar
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


def is_sized(dataset: Dataset) -> TypeGuard[SizedDataset]:
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
        self.data_file = data_file
        self.target = torch.tensor(target)
        self.seg_length = seg_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.loader = loader
        self.transform = transform
        signal = loader(data_file)
        self.num_segments = len(signal) // self.seg_length

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.loader(self.data_file)
        segment = signal[idx * self.seg_length : (idx + 1) * self.seg_length]
        stft = torch.stft(
            torch.tensor(segment), self.win_length, self.hop_length, return_complex=True
        )
        amplitude = stft.abs()
        db = 20 * amplitude.log10()
        image = self.transform(db.unsqueeze(dim=0))
        return image, self.target


class TransformDataset(Dataset):
    def __init__(
        self, dataset: Dataset, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        if is_sized(self.dataset):
            return len(self.dataset)
        raise AttributeError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        image = self.transform(image)
        return image, label


Subset = Literal["train", "valid", "test"]


def compute_mean_std(data_loader: DataLoader) -> tuple[float, float]:
    pixel_sum = 0.0
    num_pixels = 0

    for image_batch, _ in data_loader:
        pixel_sum += image_batch.sum()
        num_pixels += image_batch.numel()

    pixel_mean = pixel_sum / num_pixels

    pixel_ssd = sum(
        ((image_batch - pixel_mean) ** 2).sum().item() for image_batch, _ in data_loader
    )

    pixel_std = (pixel_ssd / num_pixels) ** 0.5
    return pixel_mean, pixel_std


def compute_min_max(data_loader: DataLoader) -> tuple[float, float]:
    pixel_min = float("inf")
    pixel_max = float("-inf")

    for image_batch, _ in data_loader:
        pixel_min = min(pixel_min, image_batch.min().item())
        pixel_max = max(pixel_max, image_batch.max().item())

    return pixel_min, pixel_max


class Pipeline(ABC):
    def __init__(self) -> None:
        self.data_dir: Path | None = None
        self.batch_size: int = 1
        self.num_workers: int = 1
        self.dataset: Dataset | None = None
        self.num_classes: int = 0
        self.subsets: dict[Subset, Dataset] = {}
        self.data_loaders: dict[Subset, DataLoader] = {}
        self.pixel_min: float = float("inf")
        self.pixel_max: float = float("-inf")
        self.pixel_mean: float = 0.0
        self.pixel_std: float = 0.0

    def p_download(self, data_dir: Path) -> Self:
        if not data_dir.exists():
            logging.info(f"Downloading data to '{data_dir}'...")
            self.download(data_dir)
        logging.info(f"Data downloaded at '{data_dir}'")
        self.data_dir = data_dir
        return self

    def p_build_dataset(
        self,
        image_size: tuple[int, int],
        seg_length: int,
        win_length: int,
        hop_length: int,
    ) -> Self:
        if self.data_dir is None:
            raise ValueError("Data isn't downloaded.")
        get_segment_stfts = functools.partial(
            SegmentSTFTs,
            seg_length=seg_length,
            win_length=win_length,
            hop_length=hop_length,
            loader=self.load_signal,
            transform=torchvision.transforms.Resize(image_size, antialias=None),
        )
        data_files = self.list_data_files(self.data_dir)
        encoder = LabelEncoder()
        labels = [self.read_label(file) for file in data_files]
        self.num_classes = len(np.unique(labels))
        targets = encoder.fit_transform(labels)
        self.dataset = ConcatDataset(
            [
                get_segment_stfts(file, target)
                for file, target in zip(data_files, targets)
            ]
        )
        return self

    def p_split_dataset(self, fractions: tuple[float, float, float]) -> Self:
        if self.dataset is None:
            raise ValueError("Dataset isn't built.")
        (
            self.subsets["train"],
            self.subsets["valid"],
            self.subsets["test"],
        ) = random_split(self.dataset, fractions)
        return self

    def p_build_data_loaders(self, batch_size: int, num_workers: int) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise ValueError("Dataset isn't built or split.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.build_data_loader("train")
        self.build_data_loader("valid")
        self.build_data_loader("test")
        return self

    def p_truncate(self, n_sigma: int) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders aren't built.")
        logging.info("Truncating data...")
        self.truncate("train", n_sigma)
        self.truncate("valid", n_sigma)
        self.truncate("test", n_sigma)
        logging.info("Data truncated")
        return self

    def p_min_max_scale(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders aren't built.")
        logging.info("Min-max scaling data...")
        self.min_max_scale("train")
        self.min_max_scale("valid")
        self.min_max_scale("test")
        logging.info("Data min-max scaled")
        return self

    def p_normalize(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders aren't built.")
        logging.info("Normalizing data...")
        self.normalize("train")
        self.normalize("valid")
        self.normalize("test")
        logging.info("Data normalized")
        return self

    def build_data_loader(self, subset: Subset) -> None:
        self.data_loaders[subset] = DataLoader(
            self.subsets[subset],
            self.batch_size,
            shuffle=(subset == "train"),
            num_workers=self.num_workers,
        )

    def truncate(self, subset: Subset, n_sigma: int) -> None:
        data_loader = self.data_loaders[subset]
        if subset == "train":
            self.pixel_mean, self.pixel_std = compute_mean_std(data_loader)
            logging.debug(f"mean={self.pixel_mean:.2f}, std={self.pixel_std:.2f}")
        outlier = self.pixel_std * n_sigma
        transform = torchvision.transforms.Lambda(
            lambda image: image.clamp(-outlier, outlier)
        )
        self.subsets[subset] = TransformDataset(self.subsets[subset], transform)
        self.build_data_loader(subset)

    def min_max_scale(self, subset: Subset) -> None:
        data_loader = self.data_loaders[subset]
        if subset == "train":
            self.pixel_min, self.pixel_max = compute_min_max(data_loader)
            logging.debug(f"min={self.pixel_min:.2f}, max={self.pixel_max:.2f}")
        loc = (self.pixel_max + self.pixel_min) / 2
        scale = (self.pixel_max - self.pixel_min) / 2
        scaler = torchvision.transforms.Normalize(loc, scale)
        self.subsets[subset] = TransformDataset(self.subsets[subset], scaler)
        self.build_data_loader(subset)

    def normalize(self, subset: Subset) -> None:
        data_loader = self.data_loaders[subset]
        if subset == "train":
            self.pixel_mean, self.pixel_std = compute_mean_std(data_loader)
            logging.debug(f"mean={self.pixel_mean:.2f}, std={self.pixel_std:.2f}")
        normalizer = torchvision.transforms.Normalize(self.pixel_mean, self.pixel_std)
        self.subsets[subset] = TransformDataset(self.subsets[subset], normalizer)
        self.build_data_loader(subset)

    @abstractmethod
    def download(self, data_dir: Path) -> None:
        pass

    @abstractmethod
    def list_data_files(self, data_dir: Path) -> list[Path]:
        pass

    @abstractmethod
    def read_label(self, data_file: Path) -> str:
        pass

    @abstractmethod
    def load_signal(self, data_file: Path) -> np.ndarray:
        pass


P = TypeVar("P", bound=type[Pipeline])
pipeline_registry: dict[str, type[Pipeline]] = {}


def register_pipeline(name: str) -> Callable[[P], P]:
    def decorator(pipeline_cls: P) -> P:
        pipeline_registry[name] = pipeline_cls
        return pipeline_cls

    return decorator


def build_pipeline(name: str) -> Pipeline:
    if name not in pipeline_registry:
        raise ValueError(f"Unregistered dataset: {name}")
    pipeline_cls = pipeline_registry[name]
    return pipeline_cls()
