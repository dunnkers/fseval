from infinite_selection import InfFS
from sklearn.base import BaseEstimator


class InfiniteSelectionEstimator(BaseEstimator):
    def fit(self, X, y):
        inf = InfFS()
        [RANKED, WEIGHT] = inf.infFS(X, y, alpha=0.5, supervision=1, verbose=1)

        self.feature_importances_ = WEIGHT
        self.ranking_ = RANKED
