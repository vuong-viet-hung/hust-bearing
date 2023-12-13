import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, TypeVar, Self

import numpy as np
import pandas as pd
import scipy
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch
import torchvision


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
        *_, spectrogram = scipy.signal.spectrogram(segment, nperseg=self.nperseg, noverlap=self.noverlap)
        image = self.transform(spectrogram)
        return image, self.label


def get_transform() -> Transform:
    return torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((64, 64), antialias=None)]
    )


class DataPipeline(ABC):
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        subset = Literal["train", "valid", "test"]
        self.datasets: dict[subset, Dataset] = {}
        self.data_loaders: dict[subset, DataLoader] = {}

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
