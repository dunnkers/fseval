from fseval.cv import CrossValidator
from fseval.cv.shuffle_split import ShuffleSplit
import pytest
import numpy as np


def test_cross_validator_split():
    cv = CrossValidator()
    with pytest.raises(NotImplementedError) as e_info:
        cv.split([[]])


def test_train_test_split():
    cv = ShuffleSplit(n_splits=1, test_size=0.5)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    split = cv.split(X)
    assert len(list(split)) == 1

    train_index, test_index = cv.get_split(X)
    assert len(train_index) == 3
    assert len(test_index) == 3
