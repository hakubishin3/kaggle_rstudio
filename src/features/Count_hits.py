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


def count_past_hits(total, train_index, test_index, n_days):
    # prepare total
    total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
    total['threshold'] = total['visitStartTime'] - datetime.timedelta(days=n_days)
    total['totals.hits'] = total['totals.hits'].astype(int)

    # narrow down the data.
    grp_result = total.groupby('fullVisitorId').size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total['fullVisitorId'].isin(more_one_visitor_id)][['fullVisitorId', 'visitStartTime', 'threshold', 'totals.hits']].copy()
    search_df = search_df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])
    import pdb; pdb.set_trace()

    max_iter = total.groupby('fullVisitorId').size().max()
    max_iter = 5
    col_list = []
    for n_lag in range(1, max_iter + 1):
        col = f'lag_{n_lag}'
        col_list.append(col)
        # shift past date
        search_df[col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(n_lag)
        # if past date > (now - n_days), replace 1.
        search_df[col] = (search_df[col] > search_df['threshold']).astype(int)
        search_df[col] = search_df[col] * search_df.groupby('fullVisitorId')['totals.hits'].shift(n_lag)

    # summary count
    search_df['sum'] = search_df[col_list].sum(axis=1)
    total = total.join(search_df[['sum']], how='outer').fillna({'count': 0})
    train_result = total.loc[train_index]['sum'].reset_index(drop=True)
    test_result = total.loc[test_index]['sum'].reset_index(drop=True)

    return train_result, test_result


class Count_past_hits(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        n_days_list = [1, 3, 7, 14, 30]
        for n_days in n_days_list:
            train_result, test_result = count_past_hits(total, train_index, test_index, n_days)
            self.train_feature[f'count_past{n_days}_hits'] = train_result
            self.test_feature[f'count_past{n_days}_hits'] = test_result
