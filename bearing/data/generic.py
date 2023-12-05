import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pathlib import Path

from bearing.data import cwru


def download(dataset: str, data_dir: Path | str) -> None:
    download_funcs = {
        'cwru': cwru.download,
    }
    download_funcs[dataset](data_dir)


def make_df(dataset: str,  data_dir: Path | str) -> pd.DataFrame:
    make_df_funcs = {
        'cwru': cwru.make_df,
    }
    return make_df_funcs[dataset](data_dir)


def split_df(df: pd.DataFrame, train_load: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.load == train_load]
    eval_df = df[df.load != train_load]
    test_df, val_df = train_test_split(eval_df, stratify=eval_df.label, test_size=0.5)
    return train_df, test_df, val_df


def make_data_loader(dataset: str, df: pd.DataFrame, batch_size: int, num_samples: int | None = None) -> DataLoader:
    make_dataloader_funcs = {
        'cwru': cwru.make_data_loader,
    }
    return make_dataloader_funcs[dataset](df, batch_size, num_samples)
