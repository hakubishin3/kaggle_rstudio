import feather
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .base import Feature
from sklearn.preprocessing import LabelEncoder
from src.models.get_folds import get_GroupKFold, get_StratifiedKFold
from joblib import Parallel, delayed
from pathos.multiprocessing import ProcessingPool as Pool


def make_total(train, test):
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_index = range(0, len(train))
    test_index = range(len(train), len(train) + len(test))
    return total, train_index, test_index


def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size() / frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0: '{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


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


def get_categorical_features_list():
        categorical_feature_list = [
            'channelGrouping',
            'device.deviceCategory',
            'device.browser',
            'device.operatingSystem',
            'trafficSource.campaign',
            'trafficSource.keyword',
            'trafficSource.medium',
            'trafficSource.referralPath',
            'trafficSource.source',
            'geoNetwork.city',
            'geoNetwork.continent',
            'geoNetwork.country',
            'geoNetwork.metro',
            'geoNetwork.networkDomain',
            'geoNetwork.region',
            'geoNetwork.subContinent',
        ]
        return categorical_feature_list


class DualCombination_TargetEncodeing(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        categorical_feature_list = get_categorical_features_list()

        # calc target
        train['target'] = np.log1p(train['totals.transactionRevenue'].fillna(0).astype('float').values)

        # make combination
        encoding_list = []
        cat_comb = list(itertools.combinations(categorical_feature_list, 2))
        for col1, col2 in cat_comb:
            comb_col_name = f'{col1}_{col2}'
            encoding_list.append(comb_col_name)
            train[comb_col_name] = train[col1].astype('str') + "_" + train[col2].astype('str')
            test[comb_col_name] = test[col1].astype('str') + "_" + test[col2].astype('str')

        # make categorical features
        folds = get_GroupKFold(train)
        tmp_path = './data/feature/tmp_DualCombination/'

        # 'max_nbytes=None' is required parameter !
        result = Parallel(n_jobs=-1, verbose=10, max_nbytes=None)(
            [delayed(calc_target_encoding)(train[[col, 'target']].copy(), test[[col]].copy(), col, folds, tmp_path) for col in encoding_list])

        # load temp file
        dfs = [pd.read_feather(f'{tmp_path}{col}_TE_train.ftr') for col in encoding_list]
        self.train_feature = pd.concat(dfs, axis=1)
        dfs = [pd.read_feather(f'{tmp_path}{col}_TE_test.ftr') for col in encoding_list]
        self.test_feature = pd.concat(dfs, axis=1)

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
