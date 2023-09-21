import os
import shutil
import urllib.request
import zipfile


def main():

    data_url = 'https://github.com/vuong-viet-hung/CWRU-1/archive/refs/heads/master.zip'
    compressed_file = 'CWRU-1-master.zip'
    repo = 'CWRU-1-master'
    src_data_root = f'{repo}/Data'
    dst_data_root = 'data'

    urllib.request.urlretrieve(data_url, compressed_file)

    with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
        zip_ref.extractall()

    os.rename(src_data_root, dst_data_root)
    os.remove(compressed_file)
    shutil.rmtree(repo)


if __name__ == '__main__':
    main()
