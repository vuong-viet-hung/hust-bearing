import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, TypeVar, Self

import numpy as np
import pandas as pd
import scipy
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch


Loader = Callable[[Path | str], np.ndarray]
Transform = Callable[[np.ndarray], torch.Tensor]


class DataFile(Dataset):
    def __init__(
        self,
        data_file: Path | str,
        label: int,
        segment_len: int,
        nperseg: int,
        noverlap: int,
        loader: Loader,
        transform: Transform,
    ) -> None:
        self.data_file = data_file
        self.label = torch.tensor(label)
        self.segment_len = segment_len
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.loader = loader
        self.transform = transform
        self.segment_len = 2048
        signal = loader(self.data_file)
        self.num_segments = len(signal) // self.segment_len

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.loader(self.data_file)
        segment = signal[idx * self.segment_len: (idx + 1) * self.segment_len]
        *_, stft = scipy.signal.stft(segment, nperseg=self.nperseg, noverlap=self.noverlap)
        amplitude = np.abs(stft)
        db = 20 * np.log10(amplitude)
        image = self.transform(db)
        return image, self.label


class NormalizeDataset(Dataset):
    def __init__(self, dataset: Dataset, normalizer: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        # noinspection PyTypeChecker
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        image = self.normalizer(image)
        return image, label


Subset = Literal["train", "valid", "test"]


class DataPipeline(ABC):
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.datasets: dict[Subset, Dataset] = {}
        self.data_loaders: dict[Subset, DataLoader] = {}

    def build_datasets(self, segment_len: int, nperseg: int, noverlap: int) -> Self:
        data_frame = self.get_data_frame()
        data_rows = (row for _, row in data_frame.iterrows())
        data_files = (row.file for row in data_rows)
        labels = (row.label for row in data_rows)
        data_files = [
            self.get_data_file(file, label, segment_len, nperseg, noverlap) for file, label in zip(data_files, labels)
        ]
        dataset = ConcatDataset(data_files)
        self.datasets["train"], self.datasets["valid"], self.datasets["test"] = random_split(dataset, [0.8, 0.1, 0.1])
        return self

    def build_data_loaders(self, batch_size: int) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.datasets.keys()):
            raise ValueError("Datasets haven't been built.")
        self.data_loaders["train"] = DataLoader(self.datasets["train"], batch_size, shuffle=True)
        self.data_loaders["valid"] = DataLoader(self.datasets["valid"], batch_size)
        self.data_loaders["test"] = DataLoader(self.datasets["test"], batch_size)
        return self

    def normalize_data_loaders(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Data loaders haven't been built.")
        self.normalize_data_loader("train")
        self.normalize_data_loader("valid")
        self.normalize_data_loader("test")
        return self

    def normalize_data_loader(self, subset: Subset) -> None:
        image_batch, _ = next(iter(self.data_loaders[subset]))
        batch_size = image_batch.shape[0]
        pixel_min = float("inf")
        pixel_max = float("-inf")
        for image_batch, _ in self.data_loaders[subset]:
            pixel_min = min(pixel_min, image_batch.min())
            pixel_max = max(pixel_max, image_batch.max())
        pixel_mean = (pixel_max + pixel_min) / 2
        pixel_std = (pixel_max - pixel_min) / 2
        normalizer = torchvision.transforms.Normalize(pixel_mean, pixel_std)
        self.datasets[subset] = NormalizeDataset(self.datasets[subset], normalizer)
        self.data_loaders[subset] = DataLoader(self.datasets[subset], batch_size)

    def validate_data_loaders(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.data_loaders.keys()):
            raise ValueError("Datasets haven't been built.")
        # Iterate over data loaders for sanity check
        for _ in itertools.chain(*self.data_loaders.values()):
            pass
        return self

    @abstractmethod
    def download_data(self) -> Self:
        pass

    @abstractmethod
    def get_data_frame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_data_file(
        self,
        data_file: Path | str,
        label: int,
        segment_len: int,
        nperseg: int,
        noverlap: int
    ) -> DataFile:
        pass


D = TypeVar('D', bound=type[DataPipeline])
data_pipeline_registry: dict[str, type[DataPipeline]] = {}


def register_data_pipeline(dataset_name: str) -> Callable[[D], D]:
    def decorator(data_pipeline_cls: D) -> D:
        data_pipeline_registry[dataset_name] = data_pipeline_cls
        return data_pipeline_cls
    return decorator


def get_data_pipeline(dataset_name: str, data_dir: str | Path) -> DataPipeline:
    if dataset_name not in data_pipeline_registry:
        raise ValueError(f"Unregistered dataset: {dataset_name!s}")
    data_pipeline_cls = data_pipeline_registry[dataset_name]
    return data_pipeline_cls(data_dir)
