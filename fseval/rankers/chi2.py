from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2


class Chi2(BaseEstimator):
    def fit(self, X, y):
        scores, _ = chi2(X, y)
        self.scores_ = scores

    @property
    def feature_importances_(self):
        return self.scores_ / sum(self.scores_)