#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from sklearn.feature_selection import chi2
import numpy as np

def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    ranking = np.argsort(-scores)
    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, chi2_ranking, 'Chi2')
