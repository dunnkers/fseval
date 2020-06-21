#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def mutual_info_ranking(X, y):
    scores = mutual_info_classif(X, y)
    ranking = np.argsort(-scores)
    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, mutual_info_ranking, 'MI')
