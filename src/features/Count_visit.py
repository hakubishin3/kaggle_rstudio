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


def count_past_visit(total, train_index, test_index, n_days):
    # prepare total
    total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
    total['threshold'] = total['visitStartTime'] - datetime.timedelta(days=n_days)

    # narrow down the data.
    grp_result = total.groupby('fullVisitorId').size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total['fullVisitorId'].isin(more_one_visitor_id)][['fullVisitorId', 'visitStartTime', 'threshold']].copy()
    search_df = search_df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])

    max_iter = total.groupby('fullVisitorId').size().max()
    col_list = []
    for n_lag in range(1, max_iter + 1):
        col = f'lag_{n_lag}'
        col_list.append(col)
        # shift past date
        search_df[col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(n_lag)
        # if past date > (now - n_days), replace 1.
        search_df[col] = (search_df[col] > search_df['threshold']).astype(int)

    # summary count
    search_df['count'] = search_df[col_list].sum(axis=1)
    total = total.join(search_df[['count']], how='outer').fillna({'count': 0})
    train_result = total.loc[train_index]['count'].reset_index(drop=True)
    test_result = total.loc[test_index]['count'].reset_index(drop=True)

    return train_result, test_result


def count_future_visit(total, train_index, test_index, n_days):
    # prepare total
    total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
    total['threshold'] = total['visitStartTime'] + datetime.timedelta(days=n_days)

    # narrow down the data.
    grp_result = total.groupby('fullVisitorId').size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total['fullVisitorId'].isin(more_one_visitor_id)][['fullVisitorId', 'visitStartTime', 'threshold']].copy()
    search_df = search_df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])

    max_iter = total.groupby('fullVisitorId').size().max()
    col_list = []
    for n_lag in range(1, max_iter + 1):
        col = f'lag_{n_lag}'
        col_list.append(col)
        # shift future date
        search_df[col] = search_df.groupby('fullVisitorId')['visitStartTime'].shift(-1 * n_lag)
        # if future date < (now + n_days), replace 1.
        search_df[col] = (search_df[col] < search_df['threshold']).astype(int)

    # summary count
    search_df['count'] = search_df[col_list].sum(axis=1)
    total = total.join(search_df[['count']], how='outer').fillna({'count': 0})
    train_result = total.loc[train_index]['count'].reset_index(drop=True)
    test_result = total.loc[test_index]['count'].reset_index(drop=True)

    return train_result, test_result


class Count_past_visit(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        n_days_list = [1, 3, 7, 14, 30]
        for n_days in n_days_list:
            train_result, test_result = count_past_visit(total, train_index, test_index, n_days)
            self.train_feature[f'count_past{n_days}_visit'] = train_result
            self.test_feature[f'count_past{n_days}_visit'] = test_result


class Count_future_visit(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        n_days_list = [1, 3, 7, 14, 30]
        for n_days in n_days_list:
            train_result, test_result = count_future_visit(total, train_index, test_index, n_days)
            self.train_feature[f'count_future{n_days}_visit'] = train_result
            self.test_feature[f'count_future{n_days}_visit'] = test_result


class Timedelta_past_visit(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
        total = total.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])
        total['shift_result'] = total.groupby('fullVisitorId')['visitStartTime'].shift(1)
        total['diff'] = total['visitStartTime'] - total['shift_result']
        total['diff'] = total['diff'].apply(lambda x: x.seconds)
        self.train_feature['Timedelta_past_visit'] = total[['diff']].loc[train_index].reset_index(drop=True).values.flatten()
        self.test_feature['Timedelta_past_visit'] = total[['diff']].loc[test_index].reset_index(drop=True).values.flatten()


class Timedelta_future_visit(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
        total = total.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])
        total['shift_result'] = total.groupby('fullVisitorId')['visitStartTime'].shift(-1)
        total['diff'] = total['shift_result'] - total['visitStartTime']
        total['diff'] = total['diff'].apply(lambda x: x.seconds)
        self.train_feature['Timedelta_future_visit'] = total[['diff']].loc[train_index].reset_index(drop=True).values.flatten()
        self.test_feature['Timedelta_future_visit'] = total[['diff']].loc[test_index].reset_index(drop=True).values.flatten()


class VisitNumber_corrected(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = []
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        total, train_index, test_index = make_total(train, test)
        total['visitStartTime'] = pd.to_datetime(total['visitStartTime'], unit='s')
        total = total.sort_values(['fullVisitorId', 'visitStartTime'], ascending=[False, True])
        total['value_1'] = 1
        total["visitNumber_corrected"] = total.groupby("fullVisitorId")["value_1"].cumsum()

        # add geta.
        # geta_summary = total.groupby('fullVisitorId')['visitNumber'].min() - 1
        # geta_summary.name = "geta"
        # total = pd.merge(total, geta_summary.reset_index(), how="outer", on="fullVisitorId")
        # total["visitNumber_corrected_addgeta"] = total["visitNumber_corrected"] + total["geta"]

        # calc diff
        total["visitNumber_diff"] = total["visitNumber"] - total["visitNumber_corrected"]

        self.train_feature['visitNumber_corrected'] = total[['visitNumber_corrected']].loc[train_index].reset_index(drop=True).values.flatten()
        self.test_feature['visitNumber_corrected'] = total[['visitNumber_corrected']].loc[test_index].reset_index(drop=True).values.flatten()
        self.train_feature['visitNumber_diff'] = total[['visitNumber_diff']].loc[train_index].reset_index(drop=True).values.flatten()
        self.test_feature['visitNumber_diff'] = total[['visitNumber_diff']].loc[test_index].reset_index(drop=True).values.flatten()
        # self.train_feature['visitNumber_corrected_addgeta'] = total[['visitNumber_corrected_addgeta']].loc[train_index].reset_index(drop=True).values.flatten()
        # self.test_feature['visitNumber_corrected_addgeta'] = total[['visitNumber_corrected_addgeta']].loc[test_index].reset_index(drop=True).values.flatten()
