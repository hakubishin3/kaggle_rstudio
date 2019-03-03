"""Run User-level Model"""
import gc
import json
import argparse
import numpy as np
import pandas as pd

from src.data.load_dataset import get_dataset_filename, load_dataset, save_dataset
from src.data.drop_columns import drop_columns
from src.data.process_geonetwork_conflict import process_geonetwork_conflict, get_preprocessed_geonetwork

from src.utils.logger_functions import get_module_logger
from src.utils.json_dump import save_json

from src.features.base import load_features
from src.features.Basic_user import Basic_user
from src.features.Categorical_user import FrequencyEncoding_user
from src.features.TimeSeries_user import TimeSeries_user

from src.models.lightgbm import LightGBM_user
from src.models.get_folds import get_GroupKFold, get_StratifiedKFold, get_KFold


feature_map = {
    'Basic_user': Basic_user,
    'FrequencyEncoding_user': FrequencyEncoding_user,
    'TimeSeries_user': TimeSeries_user
}

model_map = {
    'lightgbm': LightGBM_user
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

    # remove features (hits_pageviews_ratio)
    remove_cols_list = [col for col in x_train.columns if col.find("hits_pageviews_ratio") != -1]
    remove_cols_list.extend([col for col in x_train.columns if col.find("day_cos_Local") != -1])
    remove_cols_list.extend([col for col in x_train.columns if col.find("day_sin_Local") != -1])
    remove_cols_list.extend([col for col in x_train.columns if col.find("dayofweek_cos_Local") != -1])
    remove_cols_list.extend([col for col in x_train.columns if col.find("dayofweek_sin_Local") != -1])
    remove_cols_list.extend([col for col in x_train.columns if col.find("hour_cos_Local") != -1])
    remove_cols_list.extend([col for col in x_train.columns if col.find("hour_sin_Local") != -1])
    x_train.drop(remove_cols_list, axis=1, inplace=True)
    x_test.drop(remove_cols_list, axis=1, inplace=True)
    logger.debug(f'remove features: {remove_cols_list}')
    logger.debug(f'number of features: {x_train.shape[1]}')

    # pre-processing (add post-features)
    # logger.info('pre-processing: add post-features')
    # 後日追加予定（PCA, nonzeroなど）

    # make target
    train_index = np.sort(train['fullVisitorId'].unique())
    test_index = np.sort(test['fullVisitorId'].unique())

    train['target'] = train['totals.transactionRevenue'].fillna(0).astype('float').values
    train_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
    train_user_target = train_user_target.loc[train_index]
    y_train = np.log1p(train_user_target['target'])

    # create folds
    validation_name = config['cv']['method']
    logger.info(f'cross-validation: {validation_name}')
    if validation_name == 'GroupKFold':
        folds = get_GroupKFold(train)
    elif validation_name == 'KFold':
        folds = get_KFold(x_train)

    # train model
    model_name = config['model']['name']
    logger.info(f'train {model_name}')
    model = model_map[model_name]()
    oof_preds, sub_preds, evals_result = model.cv(
        x_train=x_train, y_train=y_train, x_test=x_test, folds=folds, params=config['model']
    )
    config.update(evals_result)

    # save predict result
    logger.info('save predict values.')
    save_path = config["dataset"]["intermediate_directory"] + f'pred_user_{args.out}.npz'
    np.savez(save_path, oof_preds=oof_preds, sub_preds=sub_preds)
    logger.debug(f'save files: {save_path}')

    # save submission-file (User-level)
    submission = pd.DataFrame({"fullVisitorId": test_index})
    submission["PredictedLogRevenue"] = sub_preds
    save_path = config['dataset']['output_directory'] + args.out + '_user.csv'
    submission.to_csv(save_path, index=False)
    logger.info(f'save submission-file. {save_path}')

    # save json file
    save_json(config, f'{args.out}_user', logger)


if __name__ == '__main__':
    main()
