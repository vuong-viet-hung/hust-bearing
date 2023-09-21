from typing import Callable

import torch
import torchvision
import numpy as np
import pandas as pd
import scipy


class LabelEncoder:

    classes = ['Normal', 'B', 'IR', 'OR']

    @classmethod
    def encode(cls, fault: str) -> int:
        return cls.classes.index(fault)
    
    @classmethod
    def decode(cls, target: int) -> str:
        return cls.classes[target]


def create_transform(
    sampling_rate: int,
    image_size: tuple[int, int]
) -> Callable[[np.ndarray, int], torch.Tensor]:

    motor_speeds = [1797, 1772, 1750, 1730]
    resize = torchvision.transforms.Resize(image_size, antialias=True)
    
    def transform(signal: np.ndarray, load: int) -> torch.Tensor:
        signal = torch.tensor(signal, dtype=torch.float)
        motor_speed = motor_speeds[load]
        n_fft = sampling_rate * 60 // motor_speed
        image = (
            torch.stft(signal, n_fft, normalized=True, return_complex=True)
            .unsqueeze(dim=0).abs()
        )
        resize_image = resize(image)
        return resize_image

    return transform


def create_target_transform() -> Callable[[str], torch.Tensor]:

    def target_transform(fault: str) -> torch.Tensor:
        target = LabelEncoder.encode(fault)
        return torch.tensor(target, dtype=torch.long)

    return target_transform


class CWRUSpectrograms(torch.utils.data.Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        transform: Callable[[np.ndarray, int], torch.Tensor],
        target_transform: Callable[[str], torch.Tensor],
    ) -> None:
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        sample = self.df.iloc[idx]
        file = sample.file
        fault = sample.fault
        load = sample.load
        signal_key = sample.signal_key
        signal_begin = sample.signal_begin
        signal_end = sample.signal_end

        signal = scipy.io.loadmat(file)[signal_key].squeeze()
        signal_sample = signal[signal_begin:signal_end]
        image = self.transform(signal_sample, load)

        target = self.target_transform(fault)

        return image, target
