from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2


class Chi2Classifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = chi2(X, y)
        self.feature_importances_ = scores
