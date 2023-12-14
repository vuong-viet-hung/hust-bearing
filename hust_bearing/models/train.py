import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from hust_bearing.data import get_data_pipeline


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--segment-len", type=int, default=2048)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=192)
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

    data_pipeline = get_data_pipeline(args.dataset_name, args.data_dir)
    (
        data_pipeline.download_data()
        .build_datasets(args.segment_len, args.nperseg, args.noverlap)
        .build_data_loaders(args.batch_size)
        .normalize_data_loaders()
    )


if __name__ == "__main__":
    main()
