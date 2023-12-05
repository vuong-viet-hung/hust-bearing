from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from bearing import data


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str,  choices=['cwru'])
    parser.add_argument('--train-load', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.data_dir is None:
        data_root_dir = Path('data')
        data_root_dir.mkdir(exist_ok=True)
        args.data_dir = data_root_dir / args.dataset

    if not args.data_dir.exists():
        data.download(args.dataset, args.data_dir)

    df = data.make_df(args.dataset, args.data_dir)
    train_df, test_df, val_df = data.split_df(df, args.train_load)
    train_dl = data.make_data_loader(args.dataset, train_df, args.batch_size, args.num_samples)
    test_dl = data.make_data_loader(args.dataset, test_df, args.batch_size, args.num_samples)
    val_dl = data.make_data_loader(args.dataset, val_df, args.batch_size, args.num_samples)


if __name__ == '__main__':
    main()
