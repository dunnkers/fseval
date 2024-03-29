from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2
from sklearn.preprocessing import minmax_scale


class Chi2Classifier(BaseEstimator):
    def fit(self, X, y):
        X = minmax_scale(X)
        scores, _ = chi2(X, y)
        self.feature_importances_ = scores
