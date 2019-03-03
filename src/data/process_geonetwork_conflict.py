import pandas as pd
import numpy as np
import feather
from pathlib import Path


def get_preprocessed_geonetwork(config: dict, dataset_type: str):
    path = config["dataset"]["intermediate_directory"]
    path += config["pre_processing"]["geonetwork_conflict"]["files"][dataset_type]

    return Path(path)


def process_geonetwork_conflict(train, test):
    # concat train and test.
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_index = range(0, len(train))
    test_index = range(len(train), len(train) + len(test))

    # extract columns
    geoNetwork_columns = [col for col in train.columns if "geoNetwork" in col]
    drop_columns = ['geoNetwork.latitude',
                    'geoNetwork.longitude',
                    'geoNetwork.cityId',
                    'geoNetwork.networkLocation',
                    'geoNetwork.networkDomain']
    geoNetwork_columns = [col for col in geoNetwork_columns if col not in drop_columns]

    for col in geoNetwork_columns:
        total[col] = total[col].where(total[col] != 'not available in demo dataset', np.nan)
        total[col] = total[col].where(total[col] != '(not set)', np.nan)
        total[col] = total[col].astype("category").cat.add_categories('N/A').fillna('N/A')

    country_part = total.groupby(
        ['geoNetwork.continent', 'geoNetwork.country',
         'geoNetwork.subContinent']).size().reset_index()

    # replace region.
    pairs = [('Casablanca', 'Grand Casablanca'), ('Colombo', 'Western Province'),
             ('Doha', 'Doha'), ('Guatemala City', 'Guatemala Department'),
             ('Hanoi', 'Hanoi'), ('Minsk', 'Minsk Region'),
             ('Nairobi', 'Nairobi County'), ('Tbilisi', 'Tbilisi')]

    for c, r in pairs:
        total.loc[(total['geoNetwork.city'] == c) &
                  (total['geoNetwork.region'] == 'N/A'), 'geoNetwork.region'] = r

    # replace country.
    most_common = total.groupby([
        'geoNetwork.city', 'geoNetwork.region'
    ])['geoNetwork.country'].apply(lambda x: x.mode()).reset_index()

    for idx, row in most_common.iterrows():
        total.loc[
            (total['geoNetwork.city'] == row['geoNetwork.city']) &
            (total['geoNetwork.region'] == row['geoNetwork.region']
             ) & ((total['geoNetwork.city'] != 'N/A') | (
                 (total['geoNetwork.region'] != 'N/A'))),
            'geoNetwork.country'] = row['geoNetwork.country']

    total.drop(
        ['geoNetwork.continent', 'geoNetwork.subContinent'],
        axis=1, inplace=True)

    country_continent = country_part[['geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.subContinent']]
    total = pd.merge(
        total,
        country_continent,
        on='geoNetwork.country',
        how='left')

    train = total.loc[train_index].reset_index(drop=True)
    test = total.loc[test_index].reset_index(drop=True)
    test.drop('totals.transactionRevenue', axis=1, inplace=True)   # drop nonexistent column
    return train, test
