import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, TypeVar
from typing_extensions import Self

import numpy as np
import scipy
import torchvision
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch


class PipelineError(Exception):
    pass


class SegmentSTFTs(Dataset):
    def __init__(
        self,
        data_file: Path,
        label: int,
        seg_length: int,
        win_length: int,
        hop_length: int,
        loader: Callable[[Path], np.ndarray],
        transform: Callable[[np.ndarray], torch.Tensor],
    ) -> None:
        self.data_file = data_file
        self.label = torch.tensor(label)
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
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.data_dir: Path | None = None
        self.dataset: Dataset | None = None
        self.subsets: dict[Subset, Dataset] = {}
        self.data_loaders: dict[Subset, DataLoader] = {}

    def p_download_data(self, data_dir: Path) -> Self:
        self.data_dir = data_dir
        if data_dir.exists():
            logging.info(f"Dataset is already downloaded to '{data_dir}'.")
            return self
        logging.info(f"Downloading dataset to '{data_dir}'...")
        self.download_data(data_dir)
        return self

    def p_build_dataset(
        self,
        image_size: tuple[int, int],
        seg_length: int,
        win_length: int,
        hop_length: int,
    ) -> Self:
        if self.data_dir is None:
            raise PipelineError("Dataset hasn't been downloaded.")
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(image_size, antialias=None),
            ]
        )
        get_segment_stfts = functools.partial(
            SegmentSTFTs,
            seg_length=seg_length,
            win_length=win_length,
            hop_length=hop_length,
            loader=self.load_signal,
            transform=transform,
        )
        data_files = self.list_data_files(self.data_dir)
        encoder = LabelEncoder()
        labels = encoder.fit_transform([self.read_label(file) for file in data_files])
        self.dataset = ConcatDataset(
            [get_segment_stfts(file, label) for file, label in zip(data_files, labels)]
        )
        return self

    def p_split_dataset(self, split_fractions: tuple[float, float, float]) -> Self:
        if self.dataset is None:
            raise PipelineError("Dataset hasn't been built.")
        (
            self.subsets["train"],
            self.subsets["valid"],
            self.subsets["test"],
        ) = random_split(self.dataset, split_fractions)
        return self

    def p_normalize_datasets(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise PipelineError("Dataset hasn't been built or split.")
        self.normalize_subset("train")
        self.normalize_subset("valid")
        self.normalize_subset("test")
        return self

    def p_build_data_loaders(self) -> Self:
        if {"train", "valid", "test"}.symmetric_difference(self.subsets.keys()):
            raise PipelineError("Dataset hasn't been built or split.")
        self.build_data_loader("train")
        self.build_data_loader("valid")
        self.build_data_loader("test")
        return self

    def normalize_subset(self, subset: Subset) -> None:
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

    def build_data_loader(self, subset: Subset) -> None:
        self.data_loaders[subset] = DataLoader(
            self.subsets[subset], self.batch_size, shuffle=(subset == "train")
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


P = TypeVar("P", bound=type[DataPipeline])
data_pipeline_registry: dict[str, type[DataPipeline]] = {}


def register_data_pipeline(dataset_name: str) -> Callable[[P], P]:
    def decorator(data_pipeline_cls: P) -> P:
        data_pipeline_registry[dataset_name] = data_pipeline_cls
        return data_pipeline_cls

    return decorator


def build_data_pipeline(dataset_name: str, batch_size: int) -> DataPipeline:
    if dataset_name not in data_pipeline_registry:
        raise ValueError(f"Unregistered dataset: '{dataset_name}'")
    data_pipeline_cls = data_pipeline_registry[dataset_name]
    return data_pipeline_cls(batch_size)
