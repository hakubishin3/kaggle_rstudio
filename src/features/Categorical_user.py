import feather
import pandas as pd
import numpy as np
from .base import Feature
from .feature_mapping import browser_mapping, operatingSystem_mapping,\
    adcontents_mapping, adnetworktype_mapping, slot_mapping,\
    campaign_mapping, source_mapping


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


def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()
    freq_col_name = '{}_Frequency'.format(col)
    freq_encoding = freq_encoding.reset_index().rename(columns={0: freq_col_name})
    return frame.merge(freq_encoding, on=col, how='left'), freq_col_name


def calc_target_encoding(train, test, col, folds, tmp_path):
    te_col_name = f'{col}_TE'
    train[te_col_name] = 0

    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        trn_data = train.iloc[trn_idx]
        encoding_value = trn_data.groupby(col)['target'].mean()
        train.set_index(col, inplace=True)
        train.iloc[val_idx, -1] = encoding_value
        train.reset_index(inplace=True)

    encoding_value = train.groupby(col)['target'].mean()
    test[te_col_name] = 0
    test.set_index(col, inplace=True)
    test.iloc[:, -1] = encoding_value
    test.reset_index(inplace=True)

    # save to temp file
    train[[te_col_name]].to_feather(f'{tmp_path}{te_col_name}_train.ftr')
    test[[te_col_name]].to_feather(f'{tmp_path}{te_col_name}_test.ftr')

    return te_col_name


class FrequencyEncoding_user(Feature):
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

        categorical_feature_list = [
            'channelGrouping',
            'device.deviceCategory',
            'device.browser',
            'device.operatingSystem',
            'geoNetwork.city',
            'geoNetwork.continent',
            'geoNetwork.country',
            'geoNetwork.metro',
            'geoNetwork.networkDomain',
            'geoNetwork.region',
            'geoNetwork.subContinent',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adContent',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.campaign',
            'trafficSource.keyword',
            'trafficSource.medium',
            'trafficSource.referralPath',
            'trafficSource.source',
        ]

        for col in categorical_feature_list:
            train[col] = train[col].astype(str)   # NaN covert to string
            test[col] = test[col].astype(str)

        total, train_range, test_range = make_total(train, test)
        total = total[categorical_feature_list + ["fullVisitorId"]]
        freq_cols_list = []
        for col in categorical_feature_list:
            total, freq_col_name = frequency_encoding(total, col)
            freq_cols_list.append(freq_col_name)

        train = total.loc[train_range].reset_index(drop=True)
        test = total.loc[test_range].reset_index(drop=True)

        stats = {'min': np.min,
                 'max': np.max,
                 'mean': np.mean,
                 'std': np.std,
                 }
        self.aggregate_feature(train, test, freq_cols_list, stats)

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
