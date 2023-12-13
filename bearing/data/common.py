from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from bearing.data.cwru import CWRUPipeline


class DataPipeline(ABC):
    def get_data_loaders(self, data_dir: Path | str) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.download_data(data_dir)
        data_frame = self.get_data_frame(data_dir)
        train_df, valid_df, test_df = self.split_data_frame(data_frame)
        train_dl = self.get_data_loader(train_df)
        valid_dl = self.get_data_loader(valid_df)
        test_dl = self.get_data_loader(test_df)
        return train_dl, valid_dl, test_dl

    @abstractmethod
    def download_data(self, data_dir: Path | str) -> None:
        pass

    @abstractmethod
    def get_data_frame(self, data_dir: Path | str) -> pd.DataFrame:
        pass

    def split_data_frame(self, data_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    def get_data_loader(self, data_frame: pd.DataFrame) -> DataLoader:
        pass


def get_data_pipeline(dataset_name: str) -> DataPipeline:
    data_pipelines = {'cwru': CWRUPipeline}
    return data_pipelines[dataset_name]
