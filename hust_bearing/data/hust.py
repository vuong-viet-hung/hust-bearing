import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy

from hust_bearing.data.core import Pipeline, register_pipeline


@register_pipeline("hust")
class HUSTBearingPipeline(Pipeline):
    def _download(self, data_dir: Path) -> None:
        download_url = (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cbv7jyx4p9-3.zip"
        )
        download_zip_file = Path(
            "HUST bearing a practical dataset for ball bearing fault diagnosis.zip"
        )
        download_extract_dir = Path(
            "HUST bearing a practical dataset for ball bearing fault diagnosis"
        )
        urllib.request.urlretrieve(download_url, download_zip_file)

        with zipfile.ZipFile(download_zip_file, "r") as zip_ref:
            zip_ref.extractall()

        download_zip_file.unlink()
        (download_extract_dir / "HUST bearing dataset").rename(data_dir)
        shutil.rmtree(download_extract_dir)

    def _list_data_files(self, data_dir: Path) -> list[Path]:
        return list(data_dir.glob("*.mat"))

    def _read_label(self, data_file: Path) -> str:
        match = re.fullmatch(
            r"""
            ([a-zA-Z]+)  # Fault
            (\d)  # Bearing
            0
            (\d)  # Load
            \.mat
            """,
            data_file.name,
            re.VERBOSE,
        )
        if match is None:
            raise ValueError(f"Invalid file name: {data_file.name}")
        return match.group(1)

    def _read_signal(self, data_file: Path) -> np.ndarray:
        data = scipy.io.loadmat(str(data_file))
        return data["data"].astype(np.float32).squeeze()
