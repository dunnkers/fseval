#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

def execute_extratrees_fs(X, y):
    estimator = ExtraTreesClassifier(n_estimators=50)
    estimator.fit(X, y)

    scores = estimator.feature_importances_

    # thresholding. threshold is score mean
    ranking = np.where(scores > np.mean(scores), scores, np.nan)
    rank_scores = np.unique(ranking[~np.isnan(ranking)])[::-1]
    
    for rank, rank_score in enumerate(rank_scores, start=1):
        ranking[ranking == rank_score] = rank

    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, execute_extratrees_fs, 'ExtraTrees')
