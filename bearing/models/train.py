from argparse import ArgumentParser
from pathlib import Path

from bearing.data import make_data_pipeline, split_df


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train-load', type=str)
    parser.add_argument('--valid-load', type=str)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--segment-len', type=int, default=2048)
    parser.add_argument('--num-train-samples', type=int, default=8000)
    parser.add_argument('--num-valid-samples', type=int, default=1000)
    parser.add_argument('--num-test-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    if args.data_dir is None:
        data_root_dir = Path('data')
        data_root_dir.mkdir(exist_ok=True)
        args.data_dir = data_root_dir / args.dataset

    pipeline = make_data_pipeline(args.dataset, args.batch_size, args.segment_len)
    pipeline.download(args.data_dir)
    df = pipeline.make_df(args.data_dir)
    train_df, valid_df, test_df = split_df(df, args.train_load, args.valid_load)
    train_dl = pipeline.make_data_loader(train_df, args.num_train_samples, shuffle=True)
    valid_dl = pipeline.make_data_loader(valid_df, args.num_valid_samples, shuffle=True)
    test_dl = pipeline.make_data_loader(test_df, args.num_test_samples)


if __name__ == '__main__':
    main()
