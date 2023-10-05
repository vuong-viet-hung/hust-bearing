import os

import matplotlib.pyplot as plt
import torch
import pandas as pd

from dataset import (
    CWRUSpectrograms,
    decode_label, 
    create_transform,
)
from models import LeNet5
from training import train


CSV_ROOT = 'csv'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64


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

        print(f'Training for {sampling_rate}_{end}')

        train_file = f'{CSV_ROOT}/{sampling_rate}_{end}_train.csv'
        val_file = f'{CSV_ROOT}/{sampling_rate}_{end}_val.csv'

        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        transform = create_transform(
            sampling_rate=12000 if sampling_rate == '12k' else 48000,
            image_size=IMAGE_SIZE,
        )

        train_ds = CWRUSpectrograms(train_df, transform)
        val_ds = CWRUSpectrograms(val_df, transform)

        os.makedirs('plots', exist_ok=True)
        plot_samples(train_ds, f'plots/{sampling_rate}_{end}.png')

        train_dl = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)
        
        model = LeNet5(
            n_classes=3 if sampling_rate == '48k' else 4,
        )

        os.makedirs('weights', exist_ok=True)
        train(
            model,
            train_dl,
            val_dl,
            n_epochs=30,
            lr=1e-3,
            saved_model=f'weights/{sampling_rate}_{end}.pth'
        )


if __name__ == '__main__':
    main()
