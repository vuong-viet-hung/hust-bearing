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
        transform: Callable[[np.ndarray], torch.Tensor],
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


class NormalizeDataset(Dataset):
    def __init__(
        self, dataset: Dataset, normalizer: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        if is_sized(self.dataset):
            return len(self.dataset)
        raise AttributeError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        image = self.normalizer(image)
        return image, label


Subset = Literal["train", "valid", "test"]


class Pipeline(ABC):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.data_dir: Path | None = None
        self.dataset: Dataset | None = None
        self.num_classes: int = 0
        self.subsets: dict[Subset, Dataset] = {}
        self.data_loaders: dict[Subset, DataLoader] = {}

    def p_download_data(self, data_dir: Path) -> Self:
        if not data_dir.exists():
            logging.info(f"Downloading data to '{data_dir}'...")
            self.download_data(data_dir)
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

    def p_normalize_datasets(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise ValueError("Dataset isn't built or split.")
        logging.info("Preprocessing data...")
        self.normalize_subset("train")
        self.normalize_subset("valid")
        self.normalize_subset("test")
        logging.info("Data preprocessed")
        return self

    def p_build_data_loaders(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise ValueError("Dataset isn't built or split.")
        self.build_data_loader("train")
        self.build_data_loader("valid")
        self.build_data_loader("test")
        return self

    def normalize_subset(self, subset: Subset) -> None:
        pixel_min = float("inf")
        pixel_max = float("-inf")
        data_loader = DataLoader(self.subsets[subset], self.batch_size, num_workers=8)
        for image_batch, _ in data_loader:
            pixel_min = min(pixel_min, image_batch.min())
            pixel_max = max(pixel_max, image_batch.max())
        loc = (pixel_max + pixel_min) / 2
        scale = (pixel_max - pixel_min) / 2
        normalizer = torchvision.transforms.Normalize(loc, scale)
        self.subsets[subset] = NormalizeDataset(self.subsets[subset], normalizer)

    def build_data_loader(self, subset: Subset) -> None:
        self.data_loaders[subset] = DataLoader(
            self.subsets[subset],
            self.batch_size,
            shuffle=(subset == "train"),
            num_workers=8,
        )

    @abstractmethod
    def download_data(self, data_dir: Path) -> None:
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


def build_pipeline(name: str, batch_size: int) -> Pipeline:
    if name not in pipeline_registry:
        raise ValueError(f"Unregistered dataset: {name}")
    pipeline_cls = pipeline_registry[name]
    return pipeline_cls(batch_size)
