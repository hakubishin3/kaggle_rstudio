import gc
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.data.load_dataset import get_dataset_filename, load_dataset, save_dataset
from src.data.drop_columns import drop_columns
from src.data.process_geonetwork_conflict import process_geonetwork_conflict, get_preprocessed_geonetwork

from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json

from src.features.base import load_features
from src.features.Basic import Basic
from src.features.Count_visit import \
    Count_past_visit, Count_future_visit, Timedelta_past_visit,\
    Timedelta_future_visit, VisitNumber_corrected
from src.features.Count_hits_pageviews import Count_past_hits_pageviews, Count_future_hits_pageviews
from src.features.DualCombination import DualCombination_TargetEncodeing

from src.models.lightgbm import LightGBM, LightGBM_user
from src.models.get_folds import get_GroupKFold, get_StratifiedKFold


feature_map = {
    'Basic': Basic,
    'Count_past_visit': Count_past_visit,
    'Count_future_visit': Count_future_visit,
    'Timedelta_past_visit': Timedelta_past_visit,
    'Timedelta_future_visit': Timedelta_future_visit,
    'VisitNumber_corrected': VisitNumber_corrected,
    'Count_past_hits_pageviews': Count_past_hits_pageviews,
    'Count_future_hits_pageviews': Count_future_hits_pageviews,
    'DualCombination_TargetEncodeing': DualCombination_TargetEncodeing
}

model_map = {
    'lightgbm': LightGBM
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/lightgbm_0.json')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--out', '-o', default='output_0')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    config = json.load(open(args.config))
    args_log = {"args": {
        "config": args.config,
        "debug_mode": args.debug,
        "force": args.force,
        "out": args.out
    }}
    config.update(args_log)

    # load dataset
    logger.info('load dataset.')
    train_path = get_dataset_filename(config, 'train')
    test_path = get_dataset_filename(config, 'test')
    train, test = load_dataset(train_path, test_path, args.debug)
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    # pre-processing (drop columns)
    logger.info('pre-processing: drop columns')
    train, test = drop_columns(train, test, logger)

    # pre-processing (geonetwork conflict)
    if config["pre_processing"]["geonetwork_conflict"]["enabled"] is True:
        logger.info('pre-processing: geonetwork conflict')
        train_path = get_preprocessed_geonetwork(config, "train")
        test_path = get_preprocessed_geonetwork(config, "test")

        if (train_path.exists() and test_path.exists() and not args.force):
            train, test = load_dataset(train_path, test_path, args.debug)
            logger.info(f'skipped and load processed data.')
        else:
            train, test = process_geonetwork_conflict(train, test)
            save_dataset(train_path, test_path, train, test)
            logger.info(f'save files: {train_path}')
            logger.info(f'save files: {test_path}')
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    # make features
    logger.info('make features')
    feature_path = config['dataset']['feature_directory']
    target_feature_map = \
        {name: key for name, key in feature_map.items()
         if name in config['features']}

    categorical_features = []
    for name, key in target_feature_map.items():
            f = key(feature_path)
            categorical_features.extend(f.categorical_features())

            if (f.train_path.exists() and f.test_path.exists() and not args.force):
                logger.info(f'{f.name} was skipped')
            else:
                f.run(train, test).save()

    # load features
    logger.info('load features')
    x_train, x_test = load_features(config)
    logger.debug(f'number of features: {x_train.shape[1]}')

    # pre-processing (remove features)
    remove_features_list = [
        'date_dayofyear', 'date_month', 'date_year', 'date_weekofyear', 'date_quarter',
        'visitStartTime', 'time_delta']
    """
    remove_features_list = [
        'date_dayofweek', 'date_dayofyear', 'date_hour', 'date_day', 'date_month', 'date_year', 'date_weekofyear', 'date_quarter',
        'visitStartTime', 'time_delta']
    """
    remove_features_list.extend([col for col in x_train.columns if col.find("hits_pageviews_ratio_per_") != -1])
    remove_features_list.extend(['hits_pageviews_ratio'])
    x_train.drop(remove_features_list, axis=1, inplace=True)
    x_test.drop(remove_features_list, axis=1, inplace=True)
    logger.debug(f'number of features: {x_train.shape[1]}')
    logger.debug(f'removed features: {remove_features_list}')

    # pre-processing (add post-features)
    # logger.info('pre-processing: add post-features')
    # 後日追加予定（PCA, nonzeroなど）

    # make target
    y_train = np.log1p(train['totals.transactionRevenue'].fillna(0).astype('float').values)

    # create folds
    validation_name = config['cv']['method']
    logger.info(f'cross-validation: {validation_name}')
    if validation_name == 'GroupKFold':
        folds = get_GroupKFold(train)
    elif validation_name == 'StratifiedKFold':
        folds = get_StratifiedKFold(train)

    # train model
    model_name = config['model']['name']
    logger.info(f'train {model_name}')
    model = model_map[model_name]()
    oof_preds, sub_preds, evals_result = model.cv(
        x_train=x_train, y_train=y_train, x_test=x_test,
        categorical_features=categorical_features, folds=folds, params=config['model']
    )
    config.update(evals_result)

    # save predict result
    logger.info('save predict values.')
    save_path = config["dataset"]["intermediate_directory"] + f'pred_session_{args.out}.npz'
    np.savez(save_path, oof_preds=oof_preds, sub_preds=sub_preds)
    logger.debug(f'save files: {save_path}')

    # save submission-file (Session-level)
    submission = pd.DataFrame({"fullVisitorId": test["fullVisitorId"].values})
    submission["PredictedLogRevenue"] = sub_preds

    if config['post_processing']['bounces_process']['enabled']:
        logger.info(f'post-processing: bounces process')
        test_index = test[test['totals.bounces'].notnull()].index.copy()
        submission.loc[test_index, 'PredictedLogRevenue'] = 0

    submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    submission.columns = ["fullVisitorId", "PredictedLogRevenue"]
    submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])

    save_path = config['dataset']['output_directory'] + args.out + '_session.csv'
    submission.to_csv(save_path, index=False)
    logger.info(f'save submission-file. {save_path}')

    # save json file
    save_json(config, args.out, logger)

    # Create user level predictions
    logger.info('Create user-level predictions.')
    x_train['predictions'] = np.expm1(oof_preds)
    x_test['predictions'] = sub_preds

    # Aggregate data at User level
    logger.info('Aggregate data at user-level.')
    x_train = x_train.join(train[['fullVisitorId']], how='inner')
    x_test = x_test.join(test[['fullVisitorId']], how='inner')
    trn_data = x_train.groupby('fullVisitorId').mean()
    sub_data = x_test.groupby('fullVisitorId').mean()

    # Create a list of predictions for each Visitor
    trn_pred_list = x_train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
        .apply(lambda df: list(df.predictions))\
        .apply(lambda x: {'pred_' + str(i): pred for i, pred in enumerate(x)})

    # Create a DataFrame with VisitorId as index
    # trn_pred_list contains dict
    # so creating a dataframe from it will expand dict values into columns
    trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
    trn_feats = trn_all_predictions.columns
    trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
    trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
    trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
    trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
    trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
    full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
    del trn_data, trn_all_predictions
    gc.collect()

    sub_pred_list = x_test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
        .apply(lambda df: list(df.predictions))\
        .apply(lambda x: {'pred_' + str(i): pred for i, pred in enumerate(x)})

    sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
    for f in trn_feats:
        if f not in sub_all_predictions.columns:
            sub_all_predictions[f] = np.nan
    sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
    sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
    sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
    sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
    sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
    sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
    del sub_data, sub_all_predictions
    gc.collect()

    train['target'] = train['totals.transactionRevenue'].fillna(0).astype('float').values
    trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
    trn_user_target = np.log1p(trn_user_target['target'])

    # train
    folds = get_GroupKFold(full_data[['totals.pageviews']].reset_index(), n_splits=5)
    logger.info(f'train {model_name}')
    model = LightGBM_user()
    oof_preds, sub_preds, evals_result = model.cv(
        x_train=full_data, y_train=trn_user_target, x_test=sub_full_data,
        folds=folds, params=config['model'],
    )
    config.update(evals_result)

    logger.info('save predict values.')
    save_path = config["dataset"]["intermediate_directory"] + f'pred_user_{args.out}.npz'
    np.savez(save_path, oof_preds=oof_preds, sub_preds=sub_preds, trn_index=full_data.index, sub_index=sub_full_data.index)
    logger.debug(f'save files: {save_path}')

    # save
    sub_full_data['PredictedLogRevenue'] = sub_preds
    save_path = config['dataset']['output_directory'] + args.out + '_user.csv'
    sub_full_data[['PredictedLogRevenue']].to_csv(save_path, index=True)
    logger.info(f'save submission-file. {save_path}')

    # save json file
    save_json(config, args.out, logger)


if __name__ == '__main__':
    main()
