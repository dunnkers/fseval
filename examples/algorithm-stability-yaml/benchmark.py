import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import hydra
import numpy as np
import pandas as pd
from fseval.config import PipelineConfig
from fseval.main import run_pipeline
from fseval.types import AbstractEstimator, AbstractMetric, Callback
from scipy.stats import norm
from sklearn.base import BaseEstimator
from skrebate import ReliefF

"""
The checkInputType and getStability functions come from the following paper:

[1] On the Stability of Feature Selection. Sarah Nogueira, Konstantinos Sechidis, Gavin Brown. 
    Journal of Machine Learning Reasearch (JMLR). 2017.
You can find a full demo using this package at:
http://htmlpreview.github.io/?https://github.com/nogueirs/JMLR2017/blob/master/python/stabilityDemo.html
NB: This package requires the installation of the packages: numpy, scipy and math
"""


def checkInputType(Z):
    """This function checks that Z is of the rigt type and dimension.
    It raises an exception if not.
    OUTPUT: The input Z as a numpy.ndarray
    """
    ### We check that Z is a list or a numpy.array
    if isinstance(Z, list):
        Z = np.asarray(Z)
    elif not isinstance(Z, np.ndarray):
        raise ValueError("The input matrix Z should be of type list or numpy.ndarray")
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim != 2:
        raise ValueError("The input matrix Z should be of dimension 2")
    return Z


def getStability(Z):
    """
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].

    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    OUTPUT: The stability of the feature selection procedure
    """
    Z = checkInputType(Z)
    M, d = Z.shape
    hatPF = np.mean(Z, axis=0)
    kbar = np.sum(hatPF)
    denom = (kbar / d) * (1 - kbar / d)
    return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom


class StabilityNogueira(AbstractMetric):
    def score_bootstrap(
        self,
        ranker: AbstractEstimator,
        validator: AbstractEstimator,
        callbacks: Callback,
        scores: Dict,
        **kwargs,
    ) -> Dict:
        # compute stability and send to table
        Z = np.array(self.support_matrix)
        Z = Z.astype(int)
        stability = getStability(Z)
        stability_df = pd.DataFrame([{"stability": stability}])
        callbacks.on_table(stability_df, "stability")

        # set in scores dict
        scores["stability"] = stability

        return scores

    def score_ranking(
        self,
        scores: Union[Dict, pd.DataFrame],
        ranker: AbstractEstimator,
        bootstrap_state: int,
        callbacks: Callback,
        feature_importances: Optional[np.ndarray] = None,
    ):
        support_matrix = getattr(self, "support_matrix", [])
        self.support_matrix = support_matrix
        self.support_matrix.append(ranker.feature_support_)


class ReliefF_FeatureSelection(ReliefF):
    def fit(self, X, y):
        super(ReliefF_FeatureSelection, self).fit(X, y)

        # extract feature subset from ReliefF
        feature_subset = self.top_features_[: self.n_features_to_select]

        # set `support_` vector
        _, p = np.shape(X)
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[feature_subset] = True


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
