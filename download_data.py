import os
import shutil
import urllib.request
import zipfile


DATA_URL = 'https://github.com/vuong-viet-hung/CWRU-1/archive/refs/heads/master.zip'
COMPRESSED_FILE = 'CWRU-1-master.zip'
REPO = 'CWRU-1-master'


def main():

    src_data_root = f'{REPO}/Data'
    dst_data_root = 'data'

    urllib.request.urlretrieve(DATA_URL, COMPRESSED_FILE)

    with zipfile.ZipFile(COMPRESSED_FILE, 'r') as zip_ref:
        zip_ref.extractall()

    os.rename(src_data_root, dst_data_root)
    os.remove(COMPRESSED_FILE)
    shutil.rmtree(REPO)


if __name__ == '__main__':
    main()
