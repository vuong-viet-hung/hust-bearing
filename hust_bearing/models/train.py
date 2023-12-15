import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn, optim

from hust_bearing import data
from hust_bearing import models


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--saved-model", type=Path)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--seg-length", type=int, default=2048)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument(
        "--split-fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1)
    )
    parser.add_argument("--seed", type=int, default=0)
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

    args.saved_model.parent.mkdir(parents=True, exist_ok=True)

    pipeline = data.build_pipeline(args.dataset_name, args.batch_size)
    (
        pipeline.p_download_data(args.data_dir)
        .p_build_dataset(
            args.image_size, args.seg_length, args.win_length, args.hop_length
        )
        .p_split_dataset(args.split_fractions)
        .p_normalize_datasets()
        .p_build_data_loaders()
    )

    model = models.LeNet5(num_classes=4, in_channels=1)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    engine = models.Engine(model, args.device, args.saved_model)
    engine.train(
        pipeline.data_loaders["train"],
        pipeline.data_loaders["valid"],
        args.num_epochs,
        loss_func,
        optimizer,
        lr_scheduler,
    )
    engine.test(pipeline.data_loaders["test"], loss_func)


if __name__ == "__main__":
    main()
