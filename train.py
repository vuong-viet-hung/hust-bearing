import matplotlib.pyplot as plt
import torch
import pandas as pd

from dataset import (
    CWRUSpectrograms,
    decode_label, 
    create_transform,
)


CSV_ROOT = 'csv'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


def plot_samples(
    dataset: torch.utils.data.Dataset, fig_name: str
):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        image, target = dataset[idx]
        fault = decode_label(target)
        ax.imshow(image.squeeze())
        ax.set_title(fault)

    fig.savefig(fig_name)


def main() -> None:

    for sampling_rate, end in zip(['12k', '12k', '48k'], ['DE', 'FE', 'DE']):

        train_file = f'{CSV_ROOT}/{sampling_rate}_{end}_train.csv'
        test_file = f'{CSV_ROOT}/{sampling_rate}_{end}_test.csv'
        val_file = f'{CSV_ROOT}/{sampling_rate}_{end}_val.csv'

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        val_df = pd.read_csv(val_file)

        transform = create_transform(
            sampling_rate=12000 if sampling_rate == '12k' else 48000,
            image_size=IMAGE_SIZE,
        )

        train_ds = CWRUSpectrograms(train_df, transform)
        test_ds = CWRUSpectrograms(test_df, transform)
        val_ds = CWRUSpectrograms(val_df, transform)

        plot_samples(train_ds, f'plots/{sampling_rate}_{end}.png')

        train_dl = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        test_dl = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    main()
