from argparse import ArgumentParser
from pathlib import Path

from bearing.data import get_data_pipeline


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    if args.data_dir is None:
        data_root_dir = Path('data')
        data_root_dir.mkdir(exist_ok=True)
        args.data_dir = data_root_dir / args.dataset

    data_pipeline = get_data_pipeline(args.dataset_name)
    data_pipeline.get_data_loaders(args.data_dir, args.batch_size)


if __name__ == '__main__':
    main()
