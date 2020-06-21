#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

def RFE_XGBoost(X, y):
    # Setup estimator
    xgboost_ensemble = XGBClassifier(max_depth=3, learning_rate=0.1,\
        n_estimators=200, silent=True, objective='binary:logistic',\
        booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,\
        max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,\
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,\
        random_state=0, seed=None, missing=None)
    # Setup FS method
    fs = RFE(estimator=xgboost_ensemble, n_features_to_select=1,\
        step=2, verbose=1)

    # Run Feature Selection
    fs.fit(X, y)

    return fs.ranking_

if __name__ == '__main__':
    run_pool(ranking_pool, RFE_XGBoost, 'RFE w/XGBoost')
