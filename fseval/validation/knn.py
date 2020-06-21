#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, validation_pool
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def KNN_5Fold(X, y):
    estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    scores = cross_val_score(estimator, X, y, cv=5, verbose=1)
    output = pd.DataFrame(data={
        'fold_no': range(1, len(scores) + 1),
        'score': scores
    })
    return output

if __name__ == '__main__':
    run_pool(validation_pool, KNN_5Fold, 'KNN_5Fold')
