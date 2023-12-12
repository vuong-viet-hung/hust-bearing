import logging
import itertools
import shutil
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torchvision
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from bearing.data.common import DataFile, DataPipeline, Transform, register_data_pipeline


def make_transform(image_size: tuple[int, int] = (64, 64)) -> Transform:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size, antialias=True),
        ]
    )


def load_signal(data_file: str | Path) -> np.ndarray:
    data = scipy.io.loadmat(str(data_file))
    *_, signal_key = (key for key in data.keys() if key.endswith('DE_time'))
    return data[signal_key].squeeze()


@register_data_pipeline('cwru')
class CWRUPipeline(DataPipeline):

    def download(self, data_dir: str | Path) -> None:
        if data_dir.exists():
            return

        repo_url = 'https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip'
        repo_zip_file = Path('CWRU-1-master.zip')
        repo_dir = Path('CWRU-1-master')
        logging.info(f'Downloading \'cwru\' dataset to {data_dir!r}...')
        urllib.request.urlretrieve(repo_url, repo_zip_file)

        with zipfile.ZipFile(repo_zip_file, 'r') as zip_ref:
            zip_ref.extractall()

        repo_zip_file.unlink()
        (repo_dir / 'Data').rename(data_dir)
        shutil.rmtree(repo_dir)

    def make_df(self, data_dir: str | Path) -> pd.DataFrame:
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
        df['sampling_rate'] = 12_000
        df['match'] = df.file.map(lambda file: file_regex.match(file.name))
        df['fault'] = df.match.map(lambda match: match.group(1))
        df['fault_size'] = df.match.map(lambda match: match.group(2))
        df['fault_location'] = df.match.map(lambda match: match.group(3))
        df['load'] = df.match.map(lambda match: match.group(4))
        df.drop(columns=['match'], inplace=True)
        df['rpm'] = df.load.map({'0': 1797, '1': 1772, '2': 1750, '3': 1730})
        encoder = LabelEncoder()
        df['label'] = encoder.fit_transform(df.fault)
        return df

    def make_data_file(self, data_row: pd.Series, segment_len: int, num_samples: int | None) -> Dataset:
        transform = make_transform()
        return DataFile(data_row, segment_len, load_signal, transform, num_samples)
