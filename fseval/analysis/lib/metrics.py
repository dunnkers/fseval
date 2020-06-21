#
# Jeroen Overschie 2020
#

import numpy as np
import pandas as pd
from scipy import interpolate

# https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
def interpolate_pr(replica):
    recalls = np.unique(replica.recall)
    def max_precision(recall): # find max precision value given recall
        return replica[replica['recall'] == recall]['precision'].max()
    precisions = list(map(max_precision, recalls))
    precision, recall = np.r_[1, precisions], np.r_[0, recalls]
    AP = np.sum(np.r_[0, np.diff(recall)] * precision)
    return precision, recall, AP

# interpolate PR curve over some recall points to obtain graphical 
# curve representation
def sample_interp_pr(precision, recall):
    fn = interpolate.interp1d(recall, precision, kind='next',\
                              bounds_error=False) # fill out of bounds with NaN's
                              # fill_value='extrapolate')
                              # extrapolating would mean finishing
                              # any curve by filling the last values
                              # when curve stops before recall == 1.0
    recalls = np.linspace(0, 1, 100)
    precisions = fn(recalls)

    # compute AP. exclude NaN's by interpolating only over real recall values.
    real_recalls = np.linspace(0, recall.max(), 100)
    real_precisions = fn(real_recalls)
    AP = np.trapz(real_precisions, real_recalls) # interpolated AP
    
    return precisions, recalls, AP

def interpolate_roc(replica):
    interp_fpr = np.linspace(0, 1, 100)
    fprs = replica.sort_values('fpr').fpr.values
    tprs = replica.sort_values('fpr').recall.values
    interp_tpr = np.interp(interp_fpr, fprs, tprs)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    auc = np.trapz(interp_tpr, interp_fpr)
    return interp_fpr, interp_fpr, auc

def compute_interpolated_metrics(replica):
    # ROC
    fpr, tpr, auc = interpolate_roc(replica)

    # PR curve
    precision, recall, AP = interpolate_pr(replica)
    precision_, recall_, _ = sample_interp_pr(precision, recall)
    return pd.DataFrame({ 'fpr': fpr, 'tpr': tpr, 'auc': auc,
                          'precision': precision_, 'recall': recall_, 'AP': AP })


"""
Function to add metrics to legend entries. `g` should be a Seaborn object.

Note theres both `hue_name` and `hue_var`, such that the legend
can be replaced by a different hue variable, e.g. an  abbreviated name 
of the original.
"""
def add_metric_to_legend(g, metric_name, hue_name, hue_var=None):
    hue_var = hue_var if hue_var else g._hue_var
    assert hue_var, 'no hue var'
    assert (hue_name and hue_var and hue_name != hue_var),\
        'hue_name and hue_var cannot be the same'

    # contruct grid with same shape as axes, store data.
    facet_data_ = np.empty(g.axes.shape, dtype=object)
    facet_data  = np.atleast_2d(facet_data_) # when only 1 row
    for coords, data in g.facet_data():
        i, j, _ = coords
        facet_data[i, j] = pd.concat([ # merge
            facet_data[i, j], data])

    # loop axes & data, attach legend.
    for ax, axdata in zip(g.axes.ravel(), facet_data.ravel()):
        agg_dict = {}
        agg_dict[metric_name] = 'mean'
        
        # groupby. sort lexicographically on hue_var, not hue_name
        metrics = axdata\
            .groupby([hue_var, hue_name])\
            .agg(agg_dict)\
            .reset_index()
        fmt = lambda metric, hue: '{:.2f} {}'.format(metric, hue)
        labels = metrics[metric_name].combine(metrics[hue_name], fmt)
        leg = ax.legend(labels, title='{}, {}'.format(
            metric_name, hue_name))
        leg._legend_box.align = 'right'