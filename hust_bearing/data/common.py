import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, Sequence, TypeVar
from typing_extensions import Self

import numpy as np
import scipy
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch


Loader = Callable[[Path | str], np.ndarray]
Transform = Callable[[np.ndarray], torch.Tensor]


def get_transform() -> Transform:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64), antialias=None),
        ]
    )


class SegmentSTFTs(Dataset):
    def __init__(
        self,
        data_file: Path | str,
        label: int,
        seg_length: int,
        win_length: int,
        hop_length: int,
        loader: Loader,
        transform: Transform,
    ) -> None:
        self.data_file = data_file
        self.label = torch.tensor(label)
        self.seg_length = seg_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.loader = loader
        self.transform = transform
        signal = loader(self.data_file)
        self.num_segments = len(signal) // self.seg_length

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.loader(self.data_file)
        segment = signal[idx * self.seg_length : (idx + 1) * self.seg_length]
        *_, stft = scipy.signal.stft(
            segment,
            nperseg=self.win_length,
            noverlap=(self.win_length - self.hop_length),
        )
        amplitude = np.abs(stft)
        db = 20 * np.log10(amplitude)
        image = self.transform(db)
        return image, self.label


class NormalizeDataset(Dataset):
    def __init__(
        self, dataset: Dataset, normalizer: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        image = self.normalizer(image)
        return image, label


Subset = Literal["train", "valid", "test"]


class DataPipeline(ABC):
    def __init__(self, data_dir: Path | str, batch_size: int) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.dataset: Dataset = ConcatDataset([])
        self.subsets: dict[Subset, Dataset] = {}
        self.data_loaders: dict[Subset, DataLoader] = {}

    def build_dataset(self, seg_length: int, win_length: int, hop_length: int) -> Self:
        data_files = self.get_data_files()
        labels = self.get_labels(data_files)
        get_segment_stfts = functools.partial(
            SegmentSTFTs,
            seg_length=seg_length,
            win_length=win_length,
            hop_length=hop_length,
            loader=self.load_signal,
            transform=get_transform(),
        )
        self.dataset = ConcatDataset(
            [get_segment_stfts(file, label) for file, label in zip(data_files, labels)]
        )
        return self

    def split_dataset(self, split_fractions: tuple[float, float, float]) -> Self:
        if len(self.dataset) == 0:  # type: ignore
            raise ValueError("Dataset hasn't been built.")
        (
            self.subsets["train"],
            self.subsets["valid"],
            self.subsets["test"],
        ) = random_split(self.dataset, split_fractions)
        return self

    def normalize_datasets(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise ValueError("Dataset hasn't been built or split.")
        self.normalize_dataset("train")
        self.normalize_dataset("valid")
        self.normalize_dataset("test")
        return self

    def normalize_dataset(self, subset: Subset) -> None:
        pixel_min = float("inf")
        pixel_max = float("-inf")
        data_loader = DataLoader(self.subsets[subset], self.batch_size)

        for image_batch, _ in data_loader:
            pixel_min = min(pixel_min, image_batch.min())
            pixel_max = max(pixel_max, image_batch.max())

        loc = (pixel_max + pixel_min) / 2
        scale = (pixel_max - pixel_min) / 2
        normalizer = torchvision.transforms.Normalize(loc, scale)
        self.subsets[subset] = NormalizeDataset(self.subsets[subset], normalizer)

    def build_data_loaders(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise ValueError("Dataset hasn't been built or split.")
        self.data_loaders["train"] = DataLoader(
            self.subsets["train"], self.batch_size, shuffle=True
        )
        self.data_loaders["valid"] = DataLoader(self.subsets["valid"], self.batch_size)
        self.data_loaders["test"] = DataLoader(self.subsets["test"], self.batch_size)
        return self

    @abstractmethod
    def download_data(self) -> Self:
        pass

    @abstractmethod
    def get_data_files(self) -> list[Path]:
        pass

    @abstractmethod
    def get_labels(self, data_files: Sequence[Path | str]) -> np.ndarray:
        pass

    @abstractmethod
    def load_signal(self, data_file: Path | str) -> np.ndarray:
        pass


D = TypeVar("D", bound=type[DataPipeline])
data_pipeline_registry: dict[str, type[DataPipeline]] = {}


def register_data_pipeline(dataset_name: str) -> Callable[[D], D]:
    def decorator(data_pipeline_cls: D) -> D:
        data_pipeline_registry[dataset_name] = data_pipeline_cls
        return data_pipeline_cls

    return decorator


def get_data_pipeline(
    dataset_name: str, data_dir: Path | str, batch_size: int
) -> DataPipeline:
    if dataset_name not in data_pipeline_registry:
        raise ValueError(f"Unregistered dataset: '{dataset_name}'")
    data_pipeline_cls = data_pipeline_registry[dataset_name]
    return data_pipeline_cls(data_dir, batch_size)
