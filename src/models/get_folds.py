import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, StratifiedKFold


def get_GroupKFold(train, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(train['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(train.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[train['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[train['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


def get_StratifiedKFold(train, n_splits=5, shuffle=True, random_state=71):
    train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].fillna(0).astype('float').values)
    grp_result = train.groupby("fullVisitorId")['totals.transactionRevenue'].sum().reset_index()
    customer_list = grp_result[grp_result['totals.transactionRevenue'] != 0]['fullVisitorId'].tolist()

    group = pd.DataFrame()
    group['fullVisitorId'] = train['fullVisitorId'].unique()
    group['customer_flg'] = 0
    customer_index_in_group = group.query('fullVisitorId in @customer_list').index
    group.loc[customer_index_in_group, 'customer_flg'] = 1

    group_skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_ids = []
    ids = np.arange(train.shape[0])
    for train_index, valid_index in group_skf.split(group.fullVisitorId, group.customer_flg):
        fold_ids.append([
            ids[train['fullVisitorId'].isin(group.iloc[train_index]['fullVisitorId'])],
            ids[train['fullVisitorId'].isin(group.iloc[valid_index]['fullVisitorId'])]
        ])

    return fold_ids


def get_KFold(train, n_splits=5, shuffle=True, random_state=71):
    folds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_ids = []

    for train_index, valid_index in folds.split(train):
        fold_ids.append([train_index, valid_index])

    return fold_ids
