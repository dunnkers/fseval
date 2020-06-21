#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, analysis_pool
import numpy as np
import pandas as pd

def compute_binary_subsets(rankdata):
    rankdata['subset' ]          = rankdata['subset'].str.split(pat=',')
    rankdata['n_positives']      = rankdata['subset'].str.len()

    def get_binary_subset(row):
        p       = int(row.p)
        subset  = np.array(row.subset).astype(int)
        binary_subset = np.zeros(p, dtype=int)
        binary_subset[subset] = 1
        return binary_subset

    rows = []
    for _, data in rankdata.groupby([ 'ranking_method', 'cpu_time' ]):
        for k in range(1, 11): # k=1, k=2, ..., k=10
            # select only results with n_features_selected <= k
            relevant_rows   = data[data['n_positives'] <= k]
            if relevant_rows.empty:
                continue
            last_ranked_idx = relevant_rows['feature_rank'].argmax()
            last_ranked_row = relevant_rows.iloc[last_ranked_idx].copy()

            # compute binary subset
            binary_subset_int   = get_binary_subset(last_ranked_row)
            binary_subset       = ','.join(binary_subset_int.astype(str))

            # store
            last_ranked_row['binary_subset'] = binary_subset
            last_ranked_row['max_features'] = k
            rows.append(last_ranked_row)

    bindata = pd.concat(rows, axis=1).transpose()
    bindata = bindata.drop(columns=[ # storage space reduction
        'subset', 'replicas', 'feature_index', 'p', 'n', 'p_informative'])
    return bindata

if __name__ == '__main__':
    run_pool(analysis_pool, compute_binary_subsets, 'binary_subsets')
