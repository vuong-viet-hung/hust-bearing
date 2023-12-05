from argparse import ArgumentParser
from pathlib import Path

from bearing.data import cwru


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    cwru.download_data(args.data_dir)
