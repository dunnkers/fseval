from fseval.types import Ranker
from sklearn.feature_selection import chi2

class Chi2(Ranker):
    def fit(self, X, y):
        scores, _ = chi2(X, y)
        self.scores_ = scores
    
    @property
    def feature_importances_(self): return self.scores_
