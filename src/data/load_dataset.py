import feather
import pandas as pd
from pathlib import Path


def get_dataset_filename(config: dict, dataset_type: str):
    path = config['dataset']['intermediate_directory']
    path += config['dataset']['files'][dataset_type]

    return Path(path)


def load_dataset(train_path, test_path, debug_mode: bool, nrows=1000):
    train = feather.read_dataframe(train_path, use_threads=-1)
    test = feather.read_dataframe(test_path, use_threads=-1)

    if debug_mode is True:
        train = train.iloc[:nrows]
        test = test.iloc[:nrows]

    return train, test


def save_dataset(train_path, test_path, train, test):
    feather.write_dataframe(train, train_path)
    feather.write_dataframe(test, test_path)
