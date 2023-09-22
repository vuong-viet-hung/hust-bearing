import glob
import os
import re

import pandas as pd
import scipy
from sklearn.model_selection import train_test_split


DATA_ROOT = 'data'
CSV_ROOT = 'csv'
FILENAME_REGEX = re.compile(
    r'''
    ([a-zA-Z]+)  # Fault
    (\d{3})?  # Fault diameter
    (@\d+)?  # Fault location
    _
    (\d+)  # Load
    \.mat
    ''',
    re.VERBOSE,
)
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42
SAMPLING_RATES = ['12k', '12k', '48k']
ENDS = ['DE', 'FE', 'DE']


def create_df(data_files, end, sample_length):
        
    data_dict = {
        'file': [],
        'fault': [],
        'fault_diameter': [],
        'fault_location': [],
        'load': [],
        'signal_key': [],
        'signal_begin': [],
        'signal_end': [],
    }

    for file in data_files:
        data = scipy.io.loadmat(file)
        filename = os.path.basename(file)
        match = FILENAME_REGEX.fullmatch(filename)
        fault = match.group(1)
        fault_diameter = (
            int(match.group(2)) 
            if match.group(2) is not None else None
        )
        fault_location = (
            int(match.group(3).lstrip('@')) 
            if match.group(3) is not None else None
        )
        load = int(match.group(4))
        *_, signal_key = (
            [key for key in data.keys() if key.endswith(f'{end}_time')]
        )
        signal = data[signal_key]
        n_samples = len(signal) // sample_length

        for idx in range(n_samples):
            signal_begin = idx * sample_length
            signal_end = (idx + 1) * sample_length
            data_dict['file'].append(file)
            data_dict['fault'].append(fault)
            data_dict['fault_diameter'].append(fault_diameter)
            data_dict['fault_location'].append(fault_location)
            data_dict['load'].append(load)
            data_dict['signal_key'].append(signal_key)
            data_dict['signal_begin'].append(signal_begin)
            data_dict['signal_end'].append(signal_end)

    df = pd.DataFrame(data_dict)

    return df


def split_df(df, test_size, val_size, random_state=None):
    eval_size = test_size + val_size
    train_df, eval_df = train_test_split(
        df, 
        test_size=eval_size, 
        random_state=random_state, 
        stratify=df.fault
    )
    test_df, val_df = train_test_split(
        eval_df, 
        test_size=val_size / eval_size,
        random_state=random_state,
        stratify=eval_df.fault,
    )
    return train_df, test_df, val_df


def main() -> None:

    normal_root = f'{DATA_ROOT}/Normal'
    normal_files = glob.glob(f'{normal_root}/*.mat')
    os.makedirs(CSV_ROOT, exist_ok=True)

    for sampling_rate, end in zip(SAMPLING_RATES, ENDS):

        fault_root = f'{DATA_ROOT}/{sampling_rate}_{end}'
        fault_files = glob.glob(f'{fault_root}/*.mat')
        data_files = normal_files + fault_files
        sample_length = 2048 if sampling_rate == '12k' else 8192
        df = create_df(data_files, end, sample_length)

        train_df, test_df, val_df = split_df(
            df, TEST_SIZE, VAL_SIZE, RANDOM_STATE
        )
        train_df.to_csv(
            f'{CSV_ROOT}/{sampling_rate}_{end}_train.csv', index=False
        )
        test_df.to_csv(
            f'{CSV_ROOT}/{sampling_rate}_{end}_test.csv', index=False
        )
        val_df.to_csv(
            f'{CSV_ROOT}/{sampling_rate}_{end}_val.csv', index=False
        )


if __name__ == '__main__':
    main()
