import numpy as np
from stability_selection import StabilitySelection as RealStabilitySelection


class StabilitySelection(RealStabilitySelection):
    def fit(self, X, y):
        super(StabilitySelection, self).fit(X, y)
        self.support_ = self.get_support()
        self.feature_importances_ = np.max(self.stability_scores_, axis=1)
