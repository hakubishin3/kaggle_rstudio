import copy
import pandas as pd
import lightgbm as lgb
import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error


class LightGBM(object):
    def fit(self, d_train: lgb.Dataset, d_valid: lgb.Dataset,
            categorical_features, params: dict):
        evals_result = {}
        model = lgb.train(
            params=params['model_params'],
            train_set=d_train,
            categorical_feature=categorical_features,
            valid_sets=[d_train, d_valid],
            valid_names=['train', 'valid'],
            evals_result=evals_result,
            **params['train_params']
        )
        return model, evals_result

    def cv(self, x_train, y_train, x_test, categorical_features: List[str],
           folds, params: dict):
        # init predictions
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])
        importances = pd.DataFrame(index=x_train.columns)
        best_iteration = 0
        cv_score_list = []

        # Run cross-validation
        n_folds = len(folds)

        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            # make Dataset object
            d_train = lgb.Dataset(
                x_train.iloc[trn_idx], label=y_train[trn_idx],
                free_raw_data=False)
            d_valid = lgb.Dataset(
                x_train.iloc[val_idx], label=y_train[val_idx],
                free_raw_data=False)

            # train model
            params_tmp = copy.deepcopy(params)
            model, evals_result = self.fit(
                d_train, d_valid, categorical_features, params_tmp
            )
            cv_score_list.append(dict(model.best_score))
            best_iteration += model.best_iteration / n_folds

            # predict out-of-fold and test
            oof_preds[val_idx] = model.predict(x_train.iloc[val_idx], num_iteration=model.best_iteration)
            oof_preds[oof_preds < 0] = 0
            _preds = model.predict(x_test[x_train.columns], num_iteration=model.best_iteration)
            _preds[_preds < 0] = 0
            sub_preds += _preds / n_folds
            # sub_preds += np.expm1(_preds) / n_folds

            # get feature importances
            importances_tmp = pd.DataFrame(
                model.feature_importance(importance_type='gain'),
                columns=[f'gain_{i_fold+1}'],
                index=model.feature_name()
            )
            importances = importances.join(importances_tmp, how='inner')

        sub_preds = np.expm1(sub_preds)
        oof_score = mean_squared_error(y_train, oof_preds)**0.5
        print(f'OOF Score (Session-level): {oof_score}')

        feature_importance = importances.mean(axis=1)
        feature_importance = feature_importance.sort_values(ascending=False).to_dict()

        train_results = {"evals_result_session": {
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "best_iteration": best_iteration,
            "n_features": len(model.feature_name()),
            "categorical_features": categorical_features,
            "feature_importance": feature_importance
        }}

        return oof_preds, sub_preds, train_results


class LightGBM_user(object):
    def fit(self, d_train: lgb.Dataset, d_valid: lgb.Dataset, params: dict):
        evals_result = {}
        model = lgb.train(
            params=params['model_params'],
            train_set=d_train,
            valid_sets=[d_train, d_valid],
            valid_names=['train', 'valid'],
            evals_result=evals_result,
            **params['train_params']
        )
        return model, evals_result

    def cv(self, x_train, y_train, x_test, folds, params: dict):
        # init predictions
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])
        importances = pd.DataFrame(index=x_train.columns)
        best_iteration = 0
        cv_score_list = []

        # Run cross-validation
        n_folds = len(folds)

        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            # make Dataset object
            d_train = lgb.Dataset(
                x_train.iloc[trn_idx], label=y_train[trn_idx],
                free_raw_data=False)
            d_valid = lgb.Dataset(
                x_train.iloc[val_idx], label=y_train[val_idx],
                free_raw_data=False)

            # train model
            params_tmp = copy.deepcopy(params)
            model, evals_result = self.fit(
                d_train, d_valid, params_tmp
            )
            cv_score_list.append(dict(model.best_score))
            best_iteration += model.best_iteration / n_folds

            # predict out-of-fold and test
            oof_preds[val_idx] = model.predict(x_train.iloc[val_idx], num_iteration=model.best_iteration)
            oof_preds[oof_preds < 0] = 0
            _preds = model.predict(x_test[x_train.columns], num_iteration=model.best_iteration)
            _preds[_preds < 0] = 0
            sub_preds += _preds / n_folds

            # get feature importances
            importances_tmp = pd.DataFrame(
                model.feature_importance(importance_type='gain'),
                columns=[f'gain_{i_fold+1}'],
                index=model.feature_name()
            )
            importances = importances.join(importances_tmp, how='inner')

        oof_score = mean_squared_error(y_train, oof_preds)**0.5
        print(f'OOF Score (User-level): {oof_score}')

        feature_importance = importances.mean(axis=1)
        feature_importance = feature_importance.sort_values(ascending=False).to_dict()

        train_results = {"evals_result_user": {
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "best_iteration": best_iteration,
            "n_features": len(model.feature_name()),
            "feature_importance": feature_importance
        }}

        return oof_preds, sub_preds, train_results
