import pandas as pd


def drop_columns(train: pd.DataFrame, test: pd.DataFrame, logger):
    drop_columns_list = [
        'socialEngagementType',
        'device.browserSize',
        'device.browserVersion',
        'device.flashVersion',
        'device.language',
        'device.mobileDeviceBranding',
        'device.mobileDeviceInfo',
        'device.mobileDeviceMarketingName',
        'device.mobileDeviceModel',
        'device.mobileInputSelector',
        'device.operatingSystemVersion',
        'device.screenColors',
        'device.screenResolution',
        'geoNetwork.cityId',
        'geoNetwork.latitude',
        'geoNetwork.longitude',
        'geoNetwork.networkLocation',
        'trafficSource.adwordsClickInfo.criteriaParameters',
        'totals.visits'
    ]
    only_train_col = ['trafficSource.campaignCode']

    train.drop(drop_columns_list, axis=1, inplace=True)
    train.drop(only_train_col, axis=1, inplace=True)
    test.drop(drop_columns_list, axis=1, inplace=True)

    logger.debug(f'drop {len(drop_columns_list+only_train_col)}columns')
    logger.debug(f'{drop_columns_list + only_train_col}')
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    return train, test
