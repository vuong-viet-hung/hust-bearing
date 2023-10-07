from typing import Callable

import torch.utils.data
import torchvision
import numpy as np
import pandas as pd
import scipy


FAULTS = ['Normal', 'B', 'IR', 'OR']
MOTOR_SPEEDS = [1797, 1772, 1750, 1730]


def encode_label(fault: str) -> torch.Tensor:
    target = FAULTS.index(fault)
    return torch.tensor(target, dtype=torch.long)


def decode_label(target: torch.Tensor) -> str:
    return FAULTS[int(target)]


def create_transform(
    sampling_rate: int,
    image_size: tuple[int, int]
) -> Callable[[np.ndarray, int], torch.Tensor]:

    resize = torchvision.transforms.Resize(image_size, antialias=True)
    
    def transform(signal: np.ndarray, load: int) -> torch.Tensor:
        signal = torch.tensor(signal, dtype=torch.float)
        motor_speed = MOTOR_SPEEDS[load]
        n_fft = sampling_rate * 60 // motor_speed
        image = (
            torch.stft(signal, n_fft, normalized=True, return_complex=True)
            .unsqueeze(dim=0).abs()
        )
        resized_image = resize(image)
        return resized_image

    return transform


class CWRUSpectrograms(torch.utils.data.Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        transform: Callable[[np.ndarray, int], torch.Tensor],
    ) -> None:
        self.df = df
        self.transform = transform

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

        target = encode_label(fault)

        return image, target
