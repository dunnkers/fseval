import numpy as np


def normalize_feature_importances(feature_importances: np.ndarray):
    """Normalized feature importances. The summation of the importances
    vector is always 1."""

    # get ranker feature importances, check whether all components > 0
    feature_importances = np.asarray(feature_importances)
    assert not (
        feature_importances < 0
    ).any(), "estimated or ground-truth feature importances must be strictly positive."

    # normalize
    feature_importances = feature_importances / sum(feature_importances)

    return feature_importances
