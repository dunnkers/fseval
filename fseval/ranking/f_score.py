#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.feature_selection import f_classif
import numpy as np

def f_score_ranking(X, y):
    scores, _ = f_classif(X, y)
    ranking = np.argsort(-scores)
    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, f_score_ranking, 'F-value')
