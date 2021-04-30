import hydra
import numpy as np
from sklearn.model_selection import ShuffleSplit

def test_train_test_split():
    cv = hydra.utils.instantiate({
        '_target_': 'sklearn.model_selection.ShuffleSplit',
        'n_splits': 5,
        'random_state': 0
    })
    assert isinstance(cv, ShuffleSplit)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    y = np.array([1, 2, 1, 2, 1, 2])
    splits = list(cv.split(X))
    assert len(splits) == 5