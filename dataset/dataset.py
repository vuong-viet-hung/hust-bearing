from typing import Callable

import torch
import torchvision
import numpy as np
import pandas as pd
import scipy


def create_transform(
    sampling_rate: int,
    image_size: tuple[int, int]
) -> Callable[[np.ndarray, int], torch.Tensor]:

    motor_speeds = [1797, 1772, 1750, 1730]

    def stft(signal: np.ndarray, load: int) -> torch.Tensor:
        signal = torch.Tensor(signal, dtype=torch.float)
        motor_speed = motor_speeds[load]
        n_fft = sampling_rate * 60 / motor_speed
        return (
            torch.stft(signal, n_fft, return_complex=True)
            .unsqueeze(dim=0).abs()
        )

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(stft),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize(),
        ]
    )
    return transform


def create_target_transform(
        classes_file: str
    ) -> Callable[[str], torch.Tensor]:

    classes = pd.read_csv(classes_file).fault

    def target_transform(fault: str) -> torch.Tensor:
        target = classes[classes == fault].index[0]
        return torch.tensor(target, dtype=torch.long)

    return target_transform


class CWRUSpectrograms(torch.utils.data.Dataset):

    def __init__(
        self, 
        samples_file: str,
        transform: Callable[[np.ndarray, int], torch.Tensor],
        target_transform: Callable[[str], torch.Tensor],
    ) -> None:
        self.df = pd.read_csv(samples_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        sample = self.df.iloc[idx]
        file = sample['file']
        fault = sample['fault']
        load = sample['load']
        signal_begin = sample['signal_begin']
        signal_end = sample['signal_end']

        signal = scipy.io.loadmat(file)
        signal_sample = signal[signal_begin:signal_end]
        image = self.transform(signal_sample, load)

        target = self.target_transform(fault)

        return image, target
