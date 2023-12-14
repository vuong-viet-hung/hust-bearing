import logging
import shutil
import re
import urllib.request
import zipfile
from pathlib import Path
from typing_extensions import Self

import numpy as np
import scipy

from hust_bearing.data.common import DataPipeline, register_data_pipeline


@register_data_pipeline("cwru")
class CWRUPipeline(DataPipeline):
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

    def list_data_files(self) -> list[Path]:
        normal_data_files = list((self.data_dir / "Normal").glob("*.mat"))
        fault_data_files = list((self.data_dir / "12k_DE").glob("*.mat"))
        return normal_data_files + fault_data_files

    def get_label(self, data_file: Path | str) -> str:
        return self.file_regex.fullmatch(Path(data_file).name).group(1)  # type: ignore

    def load_signal(self, data_file: Path | str) -> np.ndarray:
        data = scipy.io.loadmat(str(data_file))
        *_, signal_key = (key for key in data.keys() if key.endswith("DE_time"))
        return data[signal_key].astype(np.float32).squeeze()
