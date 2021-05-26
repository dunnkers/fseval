from dataclasses import dataclass

import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor as RealTabNetRegressor


@dataclass
class TabNetRegressor(RealTabNetRegressor):
    def fit(self, X, y, **kwargs):
        # ensure target is 2D
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)
        super(TabNetRegressor, self).fit(X, y, **kwargs)
