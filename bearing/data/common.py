import itertools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, TypeVar, Self

import numpy as np
import pandas as pd
import scipy
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torch
import torchvision


Loader = Callable[[Path | str], np.ndarray]
Transform = Callable[[np.ndarray], torch.Tensor]


class DataFile(Dataset):
    def __init__(self, data_row: pd.Series, loader: Loader, transform: Transform) -> None:
        self.data_file: Path = data_row.file
        self.loader = loader
        self.transform = transform
        self.segment_len = 2048
        self.label = torch.tensor(data_row.label)
        signal = loader(self.data_file)
        self.num_segments = len(signal) // self.segment_len

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.loader(self.data_file)
        segment = signal[idx * self.segment_len: (idx + 1) * self.segment_len]
        *_, spectrogram = scipy.signal.stft(segment, nperseg=512, noverlap=384)
        image = self.transform(np.abs(spectrogram))
        return image, self.label


def get_transform() -> Transform:
    return torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((64, 64), antialias=None)]
    )


class DataPipeline(ABC):
    def __init__(self, data_dir: Path | str, data_file_cls: type[DataFile] = DataFile) -> None:
        self.data_dir = Path(data_dir)
        self.data_file_cls = data_file_cls
        self.train_ds: Dataset | None = None
        self.valid_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.train_dl: DataLoader | None = None
        self.valid_dl: DataLoader | None = None
        self.test_dl: DataLoader | None = None

    def build_datasets(self) -> Self:
        data_frame = self.get_data_frame()
        loader = self.get_loader()
        transform = get_transform()
        dataset = ConcatDataset([self.data_file_cls(row, loader, transform) for _, row in data_frame.iterrows()])
        self.train_ds, self.valid_ds, self.test_ds = random_split(dataset, [0.8, 0.1, 0.1])
        logging.debug(f"Number of train samples: {len(self.train_ds)}")
        logging.debug(f"Number of valid samples: {len(self.valid_ds)}")
        logging.debug(f"Number of test samples: {len(self.test_ds)}")
        return self

    def build_data_loaders(self, batch_size: int) -> Self:
        if any(dataset is None for dataset in [self.train_ds, self.valid_ds, self.test_ds]):
            raise ValueError("Datasets haven't been built.")
        self.train_dl = DataLoader(self.train_ds, batch_size, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size)
        self.test_dl = DataLoader(self.test_ds, batch_size)
        logging.debug(f"Number of train batches: {len(self.train_dl)}")
        logging.debug(f"Number of valid batches: {len(self.valid_dl)}")
        logging.debug(f"Number of test batches: {len(self.test_dl)}")
        return self

    def validate_data_loaders(self) -> Self:
        if any(data_loader is None for data_loader in [self.train_dl, self.valid_dl, self.test_dl]):
            raise ValueError("Data loaders haven't been built.")
        data_loader = itertools.chain(self.train_dl, self.valid_dl, self.test_dl)
        image_batch, label_batch = next(iter(data_loader))
        logging.debug(f"Image batch's shape: {image_batch.shape}")
        logging.debug(f"Sample image: {image_batch[0]}")
        logging.debug(f"Label batch's shape: {label_batch.shape}")
        logging.debug(f"Sample label batch: {label_batch}")
        logging.info("Iterate over data loaders for sanity check...")
        for _ in data_loader:
            pass
        return self

    @abstractmethod
    def download_data(self) -> Self:
        pass

    @abstractmethod
    def get_data_frame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_loader(self) -> Loader:
        pass


D = TypeVar('D', bound=type[DataPipeline])
data_pipeline_registry: dict[str, type[DataPipeline]] = {}


def register_data_pipeline(dataset_name: str) -> Callable[[D], D]:
    def decorator(data_pipeline_cls: D) -> D:
        data_pipeline_registry[dataset_name] = data_pipeline_cls
        return data_pipeline_cls
    return decorator


def get_data_pipeline(dataset_name: str, data_dir: Path | str) -> DataPipeline:
    if dataset_name not in data_pipeline_registry:
        raise ValueError(f"Unregistered dataset: {dataset_name!s}")
    data_pipeline_cls = data_pipeline_registry[dataset_name]
    return data_pipeline_cls(data_dir)
