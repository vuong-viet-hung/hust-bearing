import logging
from argparse import ArgumentParser
from pathlib import Path

from bearing.data import get_data_pipeline


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--logging-level", type=str, default="info")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    data_dir: Path | None = args.data_dir
    dataset_name: str = args.dataset_name
    batch_size: int = args.batch_size
    logging_level: str = args.logging_level

    logging.basicConfig(level=getattr(logging, logging_level.upper()), format="%(message)s")

    if data_dir is None:
        data_root_dir = Path("data")
        data_root_dir.mkdir(exist_ok=True)
        data_dir = data_root_dir / dataset_name

    data_pipeline = get_data_pipeline(dataset_name, data_dir)
    (
        data_pipeline
        .download_data()
        .build_datasets()
        .build_data_loaders(batch_size)
        .validate_data_loaders()
    )


if __name__ == "__main__":
    main()
