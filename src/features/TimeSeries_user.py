import feather
import pandas as pd
import numpy as np
from .base import Feature


def make_total(train, test):
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_range = range(0, len(train))
    test_range = range(len(train), len(train) + len(test))
    return total, train_range, test_range


def find_constant_column(df):
    n_unique_sc = df.nunique()
    return n_unique_sc[n_unique_sc == 1].index.tolist()


def find_duplicate_column(df):
    colsToRemove = []
    columns = df.columns
    for i in range(len(columns) - 1):
        v = df[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(v, df[columns[j]].values):
                colsToRemove.append(columns[j])
    return list(set(colsToRemove))


def make_timeseries_feature(train, test):
    types = ['UTC', 'Local']
    timeseries_columns_list = []

    for type_ in types:
        if type_ == 'UTC':
            train_date = pd.to_datetime(train['visitStartTime'], unit='s')
            test_date = pd.to_datetime(test['visitStartTime'], unit='s')
        elif type_ == 'Local':
            train_date = train['_local_time']
            test_date = test['_local_time']

        # dayofweek: 0 ~ 6 (Monday=0, Sunday=6)
        train[f'dayofweek_sin_{type_}'] = np.sin(2 * np.pi * train_date.dt.dayofweek / 7).round(4)
        train[f'dayofweek_cos_{type_}'] = np.cos(2 * np.pi * train_date.dt.dayofweek / 7).round(4)
        test[f'dayofweek_sin_{type_}'] = np.sin(2 * np.pi * test_date.dt.dayofweek / 7).round(4)
        test[f'dayofweek_cos_{type_}'] = np.cos(2 * np.pi * test_date.dt.dayofweek / 7).round(4)
        timeseries_columns_list.extend([f'dayofweek_sin_{type_}', f'dayofweek_cos_{type_}'])

        # hour: 0 ~ 23
        train[f'hour_sin_{type_}'] = np.sin(2 * np.pi * train_date.dt.hour / 24).round(4)
        train[f'hour_cos_{type_}'] = np.cos(2 * np.pi * train_date.dt.hour / 24).round(4)
        test[f'hour_sin_{type_}'] = np.sin(2 * np.pi * test_date.dt.hour / 24).round(4)
        test[f'hour_cos_{type_}'] = np.cos(2 * np.pi * test_date.dt.hour / 24).round(4)
        timeseries_columns_list.extend([f'hour_sin_{type_}', f'hour_cos_{type_}'])

        # day: 1 ~ 31
        train[f'day_sin_{type_}'] = np.sin(2 * np.pi * (train_date.dt.day - 1) / 31).round(4)
        train[f'day_cos_{type_}'] = np.cos(2 * np.pi * (train_date.dt.day - 1) / 31).round(4)
        test[f'day_sin_{type_}'] = np.sin(2 * np.pi * (test_date.dt.day - 1) / 31).round(4)
        test[f'day_cos_{type_}'] = np.cos(2 * np.pi * (test_date.dt.day - 1) / 31).round(4)
        timeseries_columns_list.extend([f'day_sin_{type_}', f'day_cos_{type_}'])

    return train, test, timeseries_columns_list


class TimeSeries_user(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        # make primary key
        train_index = np.sort(train['fullVisitorId'].unique())
        test_index = np.sort(test['fullVisitorId'].unique())
        self.train_feature = pd.DataFrame(index=train_index)
        self.test_feature = pd.DataFrame(index=test_index)

        train, test, timeseries_columns_list = make_timeseries_feature(train, test)
        stats = {'min': np.min,
                 'max': np.max,
                 'mean': np.mean,
                 'median': np.median,
                 'std': np.std,
                 }
        self.aggregate_feature(train, test, timeseries_columns_list, stats)

        # sort index
        assert self.train_feature.shape[0] == len(train_index)
        assert self.test_feature.shape[0] == len(test_index)
        self.train_feature = self.train_feature.loc[train_index]
        self.test_feature = self.test_feature.loc[test_index]

        # drop duplicate features
        drop_cols_list = find_duplicate_column(self.train_feature)
        if len(drop_cols_list) > 0:
            self.train_feature.drop(drop_cols_list, axis=1, inplace=True)
            self.test_feature.drop(drop_cols_list, axis=1, inplace=True)
            print(f'drop duplicate columns: {drop_cols_list}')

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)

    def aggregate_feature(self, train, test, target_cols_list, stats):
        for col in target_cols_list:
            for key, stat in stats.items():
                feature_name = f'{col}_{key}'

                # calc train
                agg_result = train.dropna(subset=[col]).groupby('fullVisitorId')[col].agg(stat)
                agg_result.name = feature_name
                self.train_feature = self.train_feature.join(agg_result, how="outer")

                # calc test
                agg_result = test.dropna(subset=[col]).groupby('fullVisitorId')[col].agg(stat)
                agg_result.name = feature_name
                self.test_feature = self.test_feature.join(agg_result, how="outer")
