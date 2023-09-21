import matplotlib.pyplot as plt
import torch
import pandas as pd

from dataset import (
    CWRUSpectrograms, create_transform, create_target_transform
)


def plot_samples(
    dataset: torch.utils.data.Dataset, 
    classes: pd.DataFrame, 
    fig_name: str
):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        image, target = dataset[idx]
        label = classes.iloc[int(target), 0]
        ax.imshow(image.squeeze())
        ax.set_title(label)

    fig.savefig(fig_name)


def main() -> None:
    csv_root = 'csv'
    image_size = (64, 64)
    batch_size = 32

    for sampling_rate, end in zip(['12k', '12k', '48k'], ['DE', 'FE', 'DE']):

        train_file = f'{csv_root}/{sampling_rate}_{end}_train.csv'
        test_file = f'{csv_root}/{sampling_rate}_{end}_test.csv'
        val_file = f'{csv_root}/{sampling_rate}_{end}_val.csv'
        classes_file = f'{csv_root}/{sampling_rate}_{end}_classes.csv'

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        val_df = pd.read_csv(val_file)
        classes = pd.read_csv(classes_file, header=None)

        transform = create_transform(
            sampling_rate=12000 if sampling_rate == '12k' else 48000,
            image_size=image_size,
        )
        target_transform = create_target_transform(classes)

        train_ds = CWRUSpectrograms(train_df, transform, target_transform)
        test_ds = CWRUSpectrograms(test_df, transform, target_transform)
        val_ds = CWRUSpectrograms(val_df, transform, target_transform)

        plot_samples(train_ds, classes, f'plots/{sampling_rate}_{end}.png')

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=True)


if __name__ == '__main__':
    main()
