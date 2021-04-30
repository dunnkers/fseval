import numpy as np
from rankers import TabNetRegressionRanker


def test_tabnet_regression_ranker():
    tabnet = TabNetRegressionRanker()
    tabnet.fit(np.array([[1, 2], [1, 2]]), np.array([[1], [2]]))
    ranking = tabnet.feature_importances_
    assert len(ranking) == 2
    assert (ranking > 0).all()
    assert ranking.sum() == 1
