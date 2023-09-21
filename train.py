import matplotlib.pyplot as plt

from dataset import CWRUSpectrograms, create_transform, create_target_transform


def main() -> None:
    csv_root = 'csv'

    for sampling_rate, end in zip(['12k', '12k', '48k'], ['DE', 'FE', 'DE']):

        train_file = f'{csv_root}/{sampling_rate}_{end}_train.csv'
        test_file = f'{csv_root}/{sampling_rate}_{end}_test.csv'
        val_file = f'{csv_root}/{sampling_rate}_{end}_val.csv'
        classes_file = f'{csv_root}/{sampling_rate}_{end}_classes.csv'

        transform = create_transform(
            sampling_rate=12000 if sampling_rate == '12k' else 48000,
            image_size=(64, 64)
        )
        target_transform = create_target_transform(classes_file)

        train_ds = CWRUSpectrograms(train_file, transform, target_transform)
        test_ds = CWRUSpectrograms(test_file, transform, target_transform)
        val_ds = CWRUSpectrograms(val_file, transform, target_transform)

        image, target = train_ds[1000]
        print(f'{image.shape}\t{target}')
        


if __name__ == '__main__':
    main()
