import feather
import pandas as pd
import numpy as np
import datetime
from .base import Feature


def make_total(train, test):
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_index = range(0, len(train))
    test_index = range(len(train), len(train) + len(test))
    return total, train_index, test_index


def count_past_hits_pageviews(total, train_index, test_index, n_days):
    # prepare total
    total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
    total['threshold'] = total['visitStartTime'] - datetime.timedelta(days=n_days)
    total['totals.hits'] = total['totals.hits'].astype(int)
    total['totals.pageviews'] = total["totals.pageviews"].fillna(0).astype(int)

    # narrow down the data.
    grp_result = total.groupby('fullVisitorId').size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total['fullVisitorId'].isin(more_one_visitor_id)][
        ['fullVisitorId', 'visitStartTime', 'threshold', 'totals.hits', 'totals.pageviews']].copy()
    search_df = search_df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])

    max_iter = total.groupby('fullVisitorId').size().max()
    hits_col_list = []
    pageviews_col_list = []
    for n_lag in range(1, max_iter + 1):
        # shift hits
        hits_col = f'hits_lag_{n_lag}'
        hits_col_list.append(hits_col)
        search_df[hits_col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(n_lag)
        search_df[hits_col] = (search_df[hits_col] > search_df['threshold']).astype(int)
        search_df[hits_col] = search_df[hits_col] * search_df.groupby('fullVisitorId')['totals.hits'].shift(n_lag)
        # shift pageviews
        pageviews_col = f'pageviews_lag_{n_lag}'
        pageviews_col_list.append(pageviews_col)
        search_df[pageviews_col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(n_lag)
        search_df[pageviews_col] = (search_df[pageviews_col] > search_df['threshold']).astype(int)
        search_df[pageviews_col] = search_df[pageviews_col] * search_df.groupby('fullVisitorId')['totals.pageviews'].shift(n_lag)

    # summary count
    search_df['hits_sum'] = search_df[hits_col_list].sum(axis=1)
    search_df['pageviews_sum'] = search_df[pageviews_col_list].sum(axis=1)
    search_df['hits_pageviews_ratio'] = search_df['hits_sum'] / (search_df['pageviews_sum'] + 1)
    result_col = ['hits_sum', 'pageviews_sum', 'hits_pageviews_ratio']
    total = total.join(search_df[result_col], how='outer').fillna({'hits_sum': 0, 'pageviews_sum': 0, 'hits_pageviews_ratio': 0})
    train_result = total.loc[train_index][result_col].reset_index(drop=True)
    test_result = total.loc[test_index][result_col].reset_index(drop=True)

    return train_result, test_result


def count_future_hits_pageviews(total, train_index, test_index, n_days):
    # prepare total
    total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
    total['threshold'] = total['visitStartTime'] + datetime.timedelta(days=n_days)
    total['totals.hits'] = total['totals.hits'].astype(int)
    total['totals.pageviews'] = total["totals.pageviews"].fillna(0).astype(int)

    # narrow down the data.
    grp_result = total.groupby('fullVisitorId').size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total['fullVisitorId'].isin(more_one_visitor_id)][
        ['fullVisitorId', 'visitStartTime', 'threshold', 'totals.hits', 'totals.pageviews']].copy()
    search_df = search_df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])

    max_iter = total.groupby('fullVisitorId').size().max()
    hits_col_list = []
    pageviews_col_list = []
    for n_lag in range(1, max_iter + 1):
        # shift hits
        hits_col = f'hits_lag_{n_lag}'
        hits_col_list.append(hits_col)
        search_df[hits_col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(-1 * n_lag)
        search_df[hits_col] = (search_df[hits_col] < search_df['threshold']).astype(int)
        search_df[hits_col] = search_df[hits_col] * search_df.groupby('fullVisitorId')['totals.hits'].shift(-1 * n_lag)
        # shift pageviews
        pageviews_col = f'pageviews_lag_{n_lag}'
        pageviews_col_list.append(pageviews_col)
        search_df[pageviews_col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(-1 * n_lag)
        search_df[pageviews_col] = (search_df[pageviews_col] < search_df['threshold']).astype(int)
        search_df[pageviews_col] = search_df[pageviews_col] * search_df.groupby('fullVisitorId')['totals.pageviews'].shift(-1 * n_lag)

    # summary count
    search_df['hits_sum'] = search_df[hits_col_list].sum(axis=1)
    search_df['pageviews_sum'] = search_df[pageviews_col_list].sum(axis=1)
    search_df['hits_pageviews_ratio'] = search_df['hits_sum'] / (search_df['pageviews_sum'] + 1)
    result_col = ['hits_sum', 'pageviews_sum', 'hits_pageviews_ratio']
    total = total.join(search_df[result_col], how='outer').fillna({'hits_sum': 0, 'pageviews_sum': 0, 'hits_pageviews_ratio': 0})
    train_result = total.loc[train_index][result_col].reset_index(drop=True)
    test_result = total.loc[test_index][result_col].reset_index(drop=True)

    return train_result, test_result


class Count_past_hits_pageviews(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        n_days_list = [1, 3, 7, 14, 30]
        for n_days in n_days_list:
            train_result, test_result = count_past_hits_pageviews(total, train_index, test_index, n_days)

            for col in train_result.columns:
                self.train_feature[f'{col}_past{n_days}'] = train_result[col]
                self.test_feature[f'{col}_past{n_days}'] = test_result[col]


class Count_future_hits_pageviews(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        n_days_list = [1, 3, 7, 14, 30]
        for n_days in n_days_list:
            train_result, test_result = count_future_hits_pageviews(total, train_index, test_index, n_days)

            for col in train_result.columns:
                self.train_feature[f'{col}_future{n_days}'] = train_result[col]
                self.test_feature[f'{col}_future{n_days}'] = test_result[col]
