import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import scipy
import torchvision

from dataset import (
    decode_label, 
    create_transform,
)


CSV_ROOT = 'csv'
SPECTROGRAM_ROOT = 'spectrograms'
IMAGE_SIZE = (64, 64)


def plot_samples(dataset: torch.utils.data.Dataset, fig_name: str):

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        image, target = dataset[idx]
        fault = decode_label(target)
        ax.imshow(image.squeeze())
        ax.set_title(fault)

    fig.savefig(fig_name)


def main() -> None:

    sampling_rates = ['12k', '12k', '48k']
    ends = ['DE', 'FE', 'DE']

    for sampling_rate, end in zip(sampling_rates, ends):

        transform = create_transform(
            sampling_rate=12000 if sampling_rate == '12k' else 48000,
            image_size=IMAGE_SIZE,
        )

        subsets = ['train', 'test', 'val']

        for subset in subsets:

            df = pd.read_csv(f'{CSV_ROOT}/{sampling_rate}_{end}_{subset}.csv')

            for idx, sample in df.iterrows():
                file = sample.file
                fault = sample.fault
                load = sample.load
                signal_key = sample.signal_key
                signal_begin = sample.signal_begin
                signal_end = sample.signal_end

                signal = scipy.io.loadmat(file)[signal_key].squeeze()
                signal_sample = signal[signal_begin:signal_end]
                image = transform(signal_sample, load)

                saved_dir = (
                    f'{SPECTROGRAM_ROOT}/'
                    f'{sampling_rate}_{end}/{subset}/{fault}'
                )
                os.makedirs(saved_dir, exist_ok=True)
                torchvision.utils.save_image(image, f'{saved_dir}/{idx:04}.png')
        

if __name__ == '__main__':
    main()
