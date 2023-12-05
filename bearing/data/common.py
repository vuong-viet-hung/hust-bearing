from typing import Callable

import numpy as np
import scipy
import torch
import torchvision


Transform = Callable[[np.ndarray], torch.Tensor]


def make_transform(image_size: tuple[int, int] = (64, 64)) -> Transform:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size, antialias=True),
        ]
    )


def sample_segments(num_segments: int, num_samples: int | None) -> np.ndarray:
    num_samples = num_segments if num_samples is None else num_samples
    return np.random.choice(num_segments, num_samples, replace=False)


def compute_spectrogram(segment: np.ndarray, nperseg: int) -> np.ndarray:
    *_, spectrogram = scipy.signal.stft(segment, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return 10 * np.log10(np.abs(spectrogram))


def compute_nums_samples_per_file(num_files: int, num_samples: int) -> np.ndarray:
    remainder = num_samples % num_files
    num_samples_per_file = num_samples // num_files
    nums_samples_per_file = np.array(
        [num_samples_per_file + 1] * remainder
        + [num_samples_per_file] * (num_files - remainder)
    )
    np.random.shuffle(nums_samples_per_file)
    return nums_samples_per_file
