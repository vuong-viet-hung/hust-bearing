import shutil
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy

from hust_bearing.data.pipeline import Pipeline, register_pipeline


@register_pipeline("cwru")
class CWRUPipeline(Pipeline):
    def download_data(self, data_dir: Path) -> None:
        download_url = (
            "https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip"
        )
        download_zip_file = Path("CWRU-1-master.zip")
        download_extract_dir = Path("CWRU-1-master")
        urllib.request.urlretrieve(download_url, download_zip_file)

        with zipfile.ZipFile(download_zip_file, "r") as zip_ref:
            zip_ref.extractall()

        download_zip_file.unlink()
        (download_extract_dir / "Data").rename(data_dir)
        shutil.rmtree(download_extract_dir)

    def list_data_files(self, data_dir: Path) -> list[Path]:
        normal_data_files = list((data_dir / "Normal").glob("*.mat"))
        fault_data_files = list((data_dir / "12k_DE").glob("*.mat"))
        return normal_data_files + fault_data_files

    def read_label(self, data_file: Path) -> str:
        return re.fullmatch(  # type: ignore
            r"""
            ([a-zA-Z]+)  # Fault
            (\d{3})?  # Fault size
            (@\d+)?  # Fault location
            _
            (\d+)  # Load
            \.mat
            """,
            data_file.name,
            re.VERBOSE,
        ).group(1)

    def load_signal(self, data_file: Path) -> np.ndarray:
        data = scipy.io.loadmat(str(data_file))
        *_, signal_key = (key for key in data.keys() if key.endswith("DE_time"))
        return data[signal_key].astype(np.float32).squeeze()
