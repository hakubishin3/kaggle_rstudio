import feather
import pandas as pd
import numpy as np
from .base import Feature
from scipy.stats import skew, kurtosis, mode


def make_total(train, test):
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_range = range(0, len(train))
    test_range = range(len(train), len(train) + len(test))
    return total, train_range, test_range


def geometric_mean(x):
    return np.exp(np.log(x).mean())


def mode_(x):
    return mode(x)[0][0]


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


class Basic_user(Feature):
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

        # create feature of quantitative variable
        train['totals.hits'] = train['totals.hits'].astype(int)
        train['totals.pageviews'] = train['totals.pageviews'].astype(float)
        train['hits_pageviews_ratio'] = train['totals.hits'] / (train['totals.pageviews'] + 1)
        test['totals.hits'] = test['totals.hits'].astype(int)
        test['totals.pageviews'] = test['totals.pageviews'].astype(float)
        test['hits_pageviews_ratio'] = test['totals.hits'] / (test['totals.pageviews'] + 1)

        quantitative_variables_list = ['totals.hits', 'totals.pageviews', 'hits_pageviews_ratio']
        stats = {'min': np.min,
                 'max': np.max,
                 'mean': np.mean,
                 'median': np.median,
                 'std': np.std,
                 'sum': np.sum,
                 'skew': skew,
                 'kurtosis': kurtosis,
                 'geometric_mean': geometric_mean,
                 'mode': mode_
                 }

        for col in quantitative_variables_list:
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

        # create feature of flag variables
        train["pageviews_NaNflg"] = train['totals.pageviews'].isnull().astype(int)
        test["pageviews_NaNflg"] = test['totals.pageviews'].isnull().astype(int)
        train["device.isMobile"] = train["device.isMobile"].astype(int)   # True or False
        test["device.isMobile"] = test["device.isMobile"].astype(int)
        train["totals.bounces"] = train["totals.bounces"].fillna(0).astype(int)   # 1 or NULL
        test["totals.bounces"] = test["totals.bounces"].fillna(0).astype(int)
        train["totals.newVisits"] = train["totals.newVisits"].fillna(0).astype(int)   # 1 or NULL
        test["totals.newVisits"] = test["totals.newVisits"].fillna(0).astype(int)
        train["trafficSource.adwordsClickInfo.isVideoAd"] = \
            train["trafficSource.adwordsClickInfo.isVideoAd"].fillna(1).astype(int)   # NULL or False. NULL means True.
        test["trafficSource.adwordsClickInfo.isVideoAd"] = \
            test["trafficSource.adwordsClickInfo.isVideoAd"].fillna(1).astype(int)
        train["trafficSource.isTrueDirect"] = train["trafficSource.isTrueDirect"].fillna(0).astype(int)   # NULL or True. NULL means False.
        test["trafficSource.isTrueDirect"] = test["trafficSource.isTrueDirect"].fillna(0).astype(int)

        flag_variables_list = ['device.isMobile', "totals.bounces", "totals.newVisits", "trafficSource.isTrueDirect",
                               "trafficSource.adwordsClickInfo.isVideoAd"]
        stats = {'sum': np.sum,
                 'min': np.min,
                 'max': np.max,
                 'std': np.std,
                 }

        for col in flag_variables_list:
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

        # sort index
        assert self.train_feature.shape[0] == len(train_index)
        assert self.test_feature.shape[0] == len(test_index)
        self.train_feature = self.train_feature.loc[train_index]
        self.test_feature = self.test_feature.loc[test_index]

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)
