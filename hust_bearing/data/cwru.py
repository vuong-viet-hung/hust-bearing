import logging
import shutil
import re
import urllib.request
import zipfile
from pathlib import Path
from typing_extensions import Self

import numpy as np
import pandas as pd
import scipy
import torchvision
from sklearn.preprocessing import LabelEncoder

from hust_bearing.data.common import DataPipeline, DataFile, register_data_pipeline


def load_signal(data_file: str | Path) -> np.ndarray:
    data = scipy.io.loadmat(str(data_file))
    *_, signal_key = (key for key in data.keys() if key.endswith("DE_time"))
    return data[signal_key].astype(np.float32).squeeze()


@register_data_pipeline("cwru")
class CWRUPipeline(DataPipeline):
    def download_data(self) -> Self:
        if self.data_dir.exists():
            logging.info(f"'cwru' dataset is already downloaded to '{self.data_dir}'.")
            return self

        logging.info(f"Downloading 'cwru' dataset to '{self.data_dir}'...")
        download_url = (
            "https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip"
        )
        download_zip_file = Path("CWRU-1-master.zip")
        download_extract_dir = Path("CWRU-1-master")
        urllib.request.urlretrieve(download_url, download_zip_file)

        with zipfile.ZipFile(download_zip_file, "r") as zip_ref:
            zip_ref.extractall()

        download_zip_file.unlink()
        (download_extract_dir / "Data").rename(self.data_dir)
        shutil.rmtree(download_extract_dir)
        return self

    def get_data_frame(self) -> pd.DataFrame:
        normal_data_files = list((self.data_dir / "Normal").glob("*.mat"))
        fault_data_files = list((self.data_dir / "12k_DE").glob("*.mat"))
        data_files = normal_data_files + fault_data_files
        df = pd.DataFrame(data_files, columns=["file"])
        file_regex = re.compile(
            r"""
            ([a-zA-Z]+)  # Fault
            (\d{3})?  # Fault size
            (@\d+)?  # Fault location
            _
            (\d+)  # Load
            \.mat
            """,
            re.VERBOSE,
        )
        df["sample_rate"] = 12_000
        df["match"] = df.file.map(lambda file: file_regex.match(file.name))
        df["fault"] = df.match.map(lambda match: match.group(1))
        df["fault_size"] = df.match.map(lambda match: match.group(2))
        df["fault_location"] = df.match.map(lambda match: match.group(3))
        df["load"] = df.match.map(lambda match: match.group(4))
        df.drop(columns=["match"], inplace=True)
        df["rpm"] = df.load.map({"0": 1797, "1": 1772, "2": 1750, "3": 1730})
        encoder = LabelEncoder()
        df["label"] = encoder.fit_transform(df.fault)
        return df

    def get_data_file(
        self,
        data_file: Path | str,
        label: int,
        segment_len: int,
        nperseg: int,
        noverlap: int,
    ) -> DataFile:
        loader = load_signal
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64), antialias=None),
            ]
        )
        return DataFile(
            data_file, label, segment_len, nperseg, noverlap, loader, transform
        )
