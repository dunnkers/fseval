import numpy as np
from featboost import FeatBoostClassifier as RealFeatBoostClassifier


class FeatBoostClassifier(RealFeatBoostClassifier):
    def fit(self, X, y):
        n, p = np.shape(X)
        self.siso_ranking_size = p

        if n < 10:
            self.number_of_folds = n
        else:
            self.number_of_folds = 10

        super(FeatBoostClassifier, self).fit(X, y)

        # extract selected feature subset, mount as `support_`
        selected_subset = np.asarray(self.selected_subset_)
        selected_subset = np.unique(selected_subset)  # see amjams/FeatBoost #2
        support_ = np.full(p, False)
        support_[selected_subset] = True
        self.support_ = support_

        # extract feature importances, mount as `feature_importances_`
        importances = self.feature_importances_array_
        # remove zero rows
        importances = importances[~np.all(importances == 0, axis=1)]
        # normalize each row by its max
        importances /= np.max(importances, axis=1)[:, None]
        # sum rows and normalize again
        importances = np.sum(importances, axis=0)
        importances /= np.max(importances)
        self.feature_importances_ = importances
