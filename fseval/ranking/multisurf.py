#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, ranking_pool
from skrebate import MultiSURF
import numpy as np

def execute_MultiSURF(X, y):
    # Setup FS method
    dataset_features = np.size(X, axis=1) # rank all features
    fs = MultiSURF(n_features_to_select=dataset_features, verbose=True,\
        n_jobs=1)

    # Run Feature Selection
    fs.fit(X, y)

    # Convert ordered feature ranking to full ranking
    selected_features = fs.top_features_
    ranking = np.full(np.size(X, axis=1), np.nan) # empty array with NaN's
    for rank, selected_feature in enumerate(selected_features, start=1):
        ranking[selected_feature] = rank

    return ranking

if __name__ == '__main__':
    run_pool(ranking_pool, execute_MultiSURF, 'MultiSURF')
