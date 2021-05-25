from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, f_regression


class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


class ANOVAFValueRegressor(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_regression(X, y)
        self.feature_importances_ = scores
