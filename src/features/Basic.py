import feather
import pandas as pd
import numpy as np
from .base import Feature
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF


def make_total(train, test):
    total = pd.concat([train, test], axis=0, sort=False, ignore_index=True)
    train_range = range(0, len(train))
    test_range = range(len(train), len(train) + len(test))
    return total, train_range, test_range


def browser_mapping(x):
    browsers = ['chrome', 'safari', 'firefox', 'internet explorer', 'edge',
                'opera', 'coc coc', 'maxthon', 'iron']
    if x in browsers:
        return x.lower()
    elif ('android' in x) or ('samsung' in x) or ('mini' in x) or\
            ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or\
            ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or\
            ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'


def adcontents_mapping(x):
    if ('google' in x):
        return 'google'
    elif ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'


def source_mapping(x):
    if ('google' in x):
        return 'google'
    elif ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'


class Basic(Feature):
    @staticmethod
    def categorical_features():
        categorical_feature_list = [
            'channelGrouping',
            'device.deviceCategory',
            'device.browser',
            'device.operatingSystem',
            'trafficSource.adContent',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.campaign',
            'trafficSource.keyword',
            'trafficSource.medium',
            'trafficSource.referralPath',
            'trafficSource.source',
            'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.isTrueDirect',
            'geoNetwork.city',
            'geoNetwork.continent',
            'geoNetwork.country',
            'geoNetwork.metro',
            'geoNetwork.networkDomain',
            'geoNetwork.region',
            'geoNetwork.subContinent',
        ]
        return categorical_feature_list

    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        categorical_feature_list = self.categorical_features()

        # make categorical features
        for col in categorical_feature_list:
            le = LabelEncoder()
            train_values = list(train[col].values.astype('str'))
            test_values = list(test[col].values.astype('str'))
            total = train_values + test_values
            le.fit(total)
            self.train_feature[col] = le.transform(train_values)
            self.test_feature[col] = le.transform(test_values)

        # make device features
        self.train_feature['device.isMobile'] = train['device.isMobile'].astype(int)
        self.test_feature['device.isMobile'] = test['device.isMobile'].astype(int)
        train['device.browser'] = train['device.browser'].map(lambda x: browser_mapping(str(x).lower())).astype('str')
        train['trafficSource.adContent'] = train['trafficSource.adContent'].map(lambda x: adcontents_mapping(str(x).lower())).astype('str')
        train['trafficSource.source'] = train['trafficSource.source'].map(lambda x: source_mapping(str(x).lower())).astype('str')
        test['device.browser'] = test['device.browser'].map(lambda x: browser_mapping(str(x).lower())).astype('str')
        test['trafficSource.adContent'] = test['trafficSource.adContent'].map(lambda x: adcontents_mapping(str(x).lower())).astype('str')
        test['trafficSource.source'] = test['trafficSource.source'].map(lambda x: source_mapping(str(x).lower())).astype('str')

        # make date features
        train_date = pd.to_datetime(train['visitStartTime'], unit='s')
        test_date = pd.to_datetime(test['visitStartTime'], unit='s')
        self.train_feature['date_dayofweek'] = train_date.dt.dayofweek
        self.test_feature['date_dayofweek'] = test_date.dt.dayofweek
        self.train_feature['date_dayofyear'] = train_date.dt.dayofyear
        self.test_feature['date_dayofyear'] = test_date.dt.dayofyear
        self.train_feature['date_hour'] = train_date.dt.hour
        self.test_feature['date_hour'] = test_date.dt.hour
        self.train_feature['date_day'] = train_date.dt.day
        self.test_feature['date_day'] = test_date.dt.day
        self.train_feature['date_month'] = train_date.dt.month
        self.test_feature['date_month'] = test_date.dt.month
        self.train_feature['date_year'] = train_date.dt.year
        self.test_feature['date_year'] = test_date.dt.year
        self.train_feature['date_weekofyear'] = train_date.dt.weekofyear
        self.test_feature['date_weekofyear'] = test_date.dt.weekofyear
        self.train_feature['date_quarter'] = train_date.dt.quarter
        self.test_feature['date_quarter'] = test_date.dt.quarter

        # make time features
        self.train_feature['visitStartTime'] = train['visitStartTime']
        self.test_feature['visitStartTime'] = test['visitStartTime']
        train_visitStartTime = pd.to_datetime(train['visitStartTime'], unit='s')
        test_visitStartTime = pd.to_datetime(test['visitStartTime'], unit='s')
        train_visitId = pd.to_datetime(train['visitId'], unit='s')
        test_visitId = pd.to_datetime(test['visitId'], unit='s')
        self.train_feature['time_delta'] = (train_visitStartTime - train_visitId).dt.seconds
        self.test_feature['time_delta'] = (test_visitStartTime - test_visitId).dt.seconds

        # make session-Id features
        total, train_range, test_range = make_total(train, test)
        summary_sessionId = total.groupby('sessionId').count()['date']
        sessionId_overlap = summary_sessionId[summary_sessionId > 1].index.tolist()
        total['sessionId_overlap_flg'] = 0
        total['sessionId_overlap_flg'] = total['sessionId_overlap_flg'].where(~total['sessionId'].isin(sessionId_overlap), 1)
        self.train_feature['sessionId_overlap_flg'] = total.iloc[train_range]['sessionId_overlap_flg'].values
        self.test_feature['sessionId_overlap_flg'] = total.iloc[test_range]['sessionId_overlap_flg'].values

        # make visitNumber features
        self.train_feature['visitNumber'] = train['visitNumber']
        self.test_feature['visitNumber'] = test['visitNumber']

        # make totals features
        self.train_feature['totals.bounces'] = train['totals.bounces'].fillna(0).astype(int)
        self.test_feature['totals.bounces'] = test['totals.bounces'].fillna(0).astype(int)
        self.train_feature['totals.hits'] = train['totals.hits'].astype(int)
        self.test_feature['totals.hits'] = test['totals.hits'].astype(int)
        self.train_feature['totals.newVisits'] = train['totals.newVisits'].fillna(0).astype(int)
        self.test_feature['totals.newVisits'] = test['totals.newVisits'].fillna(0).astype(int)
        self.train_feature['totals.pageviews'] = train['totals.pageviews'].astype(float)
        self.test_feature['totals.pageviews'] = test['totals.pageviews'].astype(float)
        self.train_feature['hits_pageviews_ratio'] = train['totals.hits'].astype(int) / (train['totals.pageviews'].fillna(0).astype(int) + 1)
        self.test_feature['hits_pageviews_ratio'] = test['totals.hits'].astype(int) / (test['totals.pageviews'].fillna(0).astype(int) + 1)

        date_units_list = ['hour', 'day', 'month', 'year', 'weekofyear', 'quarter', 'dayofyear', 'dayofweek']
        stats_list = ['mean', 'median', 'sum', 'std']
        for unit_name in date_units_list:
            for stat_name in stats_list:
                self.train_feature[f'{stat_name}_hits_per_{unit_name}'] = self.train_feature.groupby([f'date_{unit_name}'])['totals.hits'].transform(stat_name)
                self.test_feature[f'{stat_name}_hits_per_{unit_name}'] = self.test_feature.groupby([f'date_{unit_name}'])['totals.hits'].transform(stat_name)
                self.train_feature[f'{stat_name}_pageviews_per_{unit_name}'] = self.train_feature.groupby([f'date_{unit_name}'])['totals.pageviews'].transform(stat_name)
                self.test_feature[f'{stat_name}_pageviews_per_{unit_name}'] = self.test_feature.groupby([f'date_{unit_name}'])['totals.pageviews'].transform(stat_name)
                self.train_feature[f'{stat_name}_hits_pageviews_ratio_per_{unit_name}'] = self.train_feature.groupby([f'date_{unit_name}'])['hits_pageviews_ratio'].transform(stat_name)
                self.test_feature[f'{stat_name}_hits_pageviews_ratio_per_{unit_name}'] = self.test_feature.groupby([f'date_{unit_name}'])['hits_pageviews_ratio'].transform(stat_name)

        # make geoNetwork features
        self.train_feature['networkDomain_unknown'] = (train['geoNetwork.networkDomain'] == 'unknown.unknown').astype(int)
        self.test_feature['networkDomain_unknown'] = (test['geoNetwork.networkDomain'] == 'unknown.unknown').astype(int)
        self.train_feature['networkDomain_notset'] = (train['geoNetwork.networkDomain'] == '(not set)').astype(int)
        self.test_feature['networkDomain_notset'] = (test['geoNetwork.networkDomain'] == '(not set)').astype(int)
        self.train_feature['networkDomain_voxilitycom'] = (train['geoNetwork.networkDomain'] == 'voxility.com').astype(int)
        self.test_feature['networkDomain_voxilitycom'] = (test['geoNetwork.networkDomain'] == 'voxility.com').astype(int)

        self.train_feature.reset_index(drop=True, inplace=True)
        self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    train = feather.read_dataframe('../../data/interim/train.ftr')
    test = feather.read_dataframe('../../data/interim/test.ftr')
    feature_path = '../../data/feature/'
    f = Basic(feature_path)
    f.run(train, test).save()
    print(f'make features: {f.train_feature.columns}')
