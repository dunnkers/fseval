#!/usr/bin/env python
from ..pipeline.compute_pool import run_pool, analysis_pool
import numpy as np

def compute_rocdata(rankdata):
    if not 'p_informative' in rankdata.columns: # no ground truth
        return rankdata

    statdata = rankdata.dropna(subset=['p_informative'])
    if statdata.empty: # no ground truth
        return statdata

    statdata['subset' ]          = statdata['subset'].str.split(pat=',')
    statdata['p_informative']    = statdata['p_informative'].astype(str)\
                                                     .str.split(pat=',')
    n_p_informative              = statdata['p_informative'].str.len()
    n_positives                  = statdata['subset'].str.len()

    compute_tp = lambda x: np.sum(np.isin(x.subset, x.p_informative))
    tp = statdata.apply(compute_tp, axis='columns')

    fp                      = n_positives - tp
    # fn          = n_p_informative - tp
    tn                      = statdata['p'].astype(int) - n_positives
    statdata['recall']      = tp / n_p_informative
    statdata['precision']   = tp / n_positives
    statdata['fpr']         = fp / (fp + tn)
    statdata['f1']          = 2 * ((statdata['precision'] * statdata['recall'])\
                                /  (statdata['precision'] + statdata['recall']))

    statdata = statdata.drop(columns=[ # storage space reduction
        'subset', 'replicas', 'feature_index', 'p', 'n', 'p_informative'])
    return statdata

if __name__ == '__main__':
    run_pool(analysis_pool, compute_rocdata, 'roc')
