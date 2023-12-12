from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


LoadSignal = Callable[[Path | str], np.ndarray]
Transform = Callable[[np.ndarray], torch.Tensor]


class DataFile(Dataset):

    def __init__(
        self,
        data_row: pd.Series,
        segment_len: int,
        load_signal: LoadSignal,
        transform: Transform,
        num_samples: int | None
    ) -> None:
        self.data_row = data_row
        self.segment_len = segment_len
        self.load_signal = load_signal
        self.transform = transform
        self.num_samples = num_samples
        signal = self.load_signal(data_row.file)
        num_segments = signal.size - segment_len + 1
        self.sample_indices = sample_segments(num_segments, num_samples)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.load_signal(self.data_row.file)
        sample_idx = self.sample_indices[idx]
        segment = signal[sample_idx:sample_idx + self.segment_len]
        nperseg = self.data_row.sampling_rate * 60 // self.data_row.rpm
        spectrogram = compute_spectrogram(segment, nperseg)
        image = self.transform(spectrogram)
        return image, torch.tensor(self.data_row.label)


class DataPipeline(ABC):
    def __init__(self, batch_size: int, segment_len: int) -> None:
        self.batch_size = batch_size
        self.segment_len = segment_len

    @abstractmethod
    def download(self, data_dir: str | Path) -> None:
        pass

    @abstractmethod
    def make_df(self, data_dir: str | Path) -> pd.DataFrame:
        pass

    @abstractmethod
    def make_data_file(
        self,
        data_row: pd.Series,
        segment_len: int,
        num_samples: int | None,
    ) -> Dataset:
        pass

    def make_data_loader(self, df: pd.DataFrame, num_samples: int | None, shuffle: bool = False) -> DataLoader:
        nums_samples_per_file = compute_nums_samples_each_file(df.label, num_samples)
        dataset = ConcatDataset(
            [
                self.make_data_file(row, self.segment_len, num_samples=num_samples)
                for (idx, row), num_samples in zip(df.iterrows(), nums_samples_per_file)
            ]
        )
        return DataLoader(dataset, self.batch_size, shuffle=shuffle)


data_pipeline_registry: dict[str, type[DataPipeline]] = {}


def register_data_pipeline(dataset: str):
    def decorator(data_pipeline_cls: type[DataPipeline]):
        data_pipeline_registry[dataset] = data_pipeline_cls
        return data_pipeline_cls

    return decorator


def make_data_pipeline(dataset: str, batch_size: int, segment_len: int) -> DataPipeline:
    data_pipeline_cls = data_pipeline_registry[dataset]
    return data_pipeline_cls(batch_size, segment_len)


def split_df(df: pd.DataFrame, train_load: str, valid_load: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.load == train_load]
    valid_df = df[df.load == valid_load]
    test_df = df.drop(train_df.index).drop(valid_df.index)
    return train_df, valid_df, test_df


def sample_segments(num_segments: int, num_samples: int | None) -> np.ndarray:
    num_samples = num_segments if num_samples is None else num_samples
    return np.random.choice(num_segments, num_samples, replace=False)


def compute_spectrogram(segment: np.ndarray, nperseg: int) -> np.ndarray:
    *_, spectrogram = scipy.signal.stft(segment, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return 10 * np.log10(np.abs(spectrogram))


def compute_nums_samples_each_file(labels: pd.Series, num_samples: int) -> pd.Series:
    num_classes = labels.nunique()
    remainder = num_samples % num_classes
    num_samples_per_classes = num_samples // num_classes
    nums_samples_each_classes = pd.Series(
        [num_samples_per_classes + 1] * remainder
        + [num_samples_per_classes] * (num_classes - remainder)
    ).sample(frac=1.0)
    return pd.Series(
        [

        ]
    )
