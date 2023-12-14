import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from hust_bearing.data import get_data_pipeline


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seg-length", type=int, default=2048)
    parser.add_argument("--win-length", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--logging-level", type=str, default="info")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=getattr(logging, args.logging_level.upper()), format="%(message)s"
    )

    if args.data_dir is None:
        data_root_dir = Path("data")
        data_root_dir.mkdir(exist_ok=True)
        args.data_dir = data_root_dir / args.dataset_name

    data_pipeline = get_data_pipeline(args.dataset_name, args.batch_size)
    (
        data_pipeline.download_data(args.data_dir)
        .build_dataset(args.seg_length, args.win_length, args.hop_length)
        .split_dataset((0.8, 0.1, 0.1))
        .normalize_datasets()
        .build_data_loaders()
    )


if __name__ == "__main__":
    main()
