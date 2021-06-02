from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class MutualInfoClassifier(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y)
        self.feature_importances_ = scores


class MutualInfoRegressor(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_regression(X, y)
        self.feature_importances_ = scores
