import itertools
import shutil
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import scipy
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from bearing.data.common import (
    Transform, make_transform, sample_segments, compute_spectrogram, compute_nums_samples_per_file
)


def download(data_dir: str | Path) -> None:

    repo_url = 'https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip'
    repo_zip_file = Path('CWRU-1-master.zip')
    repo_dir = Path('CWRU-1-master')

    urllib.request.urlretrieve(repo_url, repo_zip_file)

    with zipfile.ZipFile(repo_zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    repo_zip_file.unlink()
    (repo_dir / 'Data').rename(data_dir)
    shutil.rmtree(repo_dir)


def make_df(data_dir: str | Path) -> pd.DataFrame:

    data_dir = Path(data_dir)
    normal_data_files = (data_dir / 'Normal').glob('*.mat')
    fault_data_files = (data_dir / '12k_DE').glob('*.mat')
    data_files = itertools.chain(normal_data_files, fault_data_files)
    df = pd.DataFrame(data_files, columns=['file'])

    file_regex = re.compile(
        r'''
        ([a-zA-Z]+)  # Fault
        (\d{3})?  # Fault size
        (@\d+)?  # Fault location
        _
        (\d+)  # Load
        \.mat
        ''',
        re.VERBOSE,
    )
    df['match'] = df.file.map(lambda file: file_regex.match(file.name))
    df['fault'] = df.match.map(lambda match: match.group(1))
    df['fault_size'] = df.match.map(lambda match: match.group(2))
    df['fault_location'] = df.match.map(lambda match: match.group(3))
    df['load'] = df.match.map(lambda match: match.group(4))
    df.drop(columns=['match'], inplace=True)

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df.fault)
    return df


class CWRUDataFile(Dataset):

    rpms = {'0': 1797, '1': 1772, '2': 1750, '3': 1730}

    def __init__(
        self,
        data_row: pd.Series,
        segment_len: int = 2048,
        transform: Transform = make_transform(),
        num_samples: int | None = None,
    ) -> None:
        self.data_row = data_row
        self.segment_len = segment_len
        self.transform = transform
        data = scipy.io.loadmat(str(data_row.file))
        *_, self.signal_key = (key for key in data.keys() if key.endswith('DE_time'))
        signal = data[self.signal_key].squeeze()
        num_segments = signal.size - segment_len + 1
        self.sample_indices = sample_segments(num_segments, num_samples)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = self.sample_indices[idx]
        data = scipy.io.loadmat(str(self.data_row.file))
        signal = data[self.signal_key].squeeze()
        segment = signal[sample_idx:sample_idx + self.segment_len]
        nperseg = 12_000 * 60 // self.rpms[self.data_row.load]
        spectrogram = compute_spectrogram(segment, nperseg)
        image = self.transform(spectrogram)
        return image, torch.tensor(self.data_row.label)


def make_data_loader(df: pd.DataFrame, batch_size: int, num_samples: int | None = None) -> DataLoader:
    nums_samples_per_file = compute_nums_samples_per_file(len(df), num_samples)
    dataset = ConcatDataset(
        [
            CWRUDataFile(row, num_samples=num_samples)
            for (idx, row), num_samples in zip(df.iterrows(), nums_samples_per_file)
        ]
    )
    return DataLoader(dataset, batch_size)
