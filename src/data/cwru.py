import shutil
import urllib.request
import zipfile
from pathlib import Path


REPO_URL = 'https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip'
REPO_ZIP_FILE = Path('CWRU-1-master.zip')
REPO_DIR = Path('CWRU-1-master')


def download_data(data_dir: str | Path) -> None:

    urllib.request.urlretrieve(REPO_URL, REPO_ZIP_FILE)

    with zipfile.ZipFile(REPO_ZIP_FILE, mode='r') as zip_ref:
        zip_ref.extractall()

    REPO_ZIP_FILE.unlink()
    (REPO_DIR / 'Data').rename(data_dir)
    shutil.rmtree(REPO_DIR)
