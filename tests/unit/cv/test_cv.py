import numpy as np
import pytest
from fseval.cv import CrossValidator
from sklearn.model_selection import ShuffleSplit


@pytest.fixture
def cv():
    return CrossValidator()


def test_no_splitter(cv):
    with pytest.raises(AssertionError):
        cv.split([[]])


def test_train_test_split(cv):
    cv.splitter = ShuffleSplit(n_splits=1, test_size=0.5)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])

    # `split()` should return a generator containing all the splits.
    split = cv.split(X)
    assert len(list(split)) == 1

    # because `test_size=0.5`, the first split (fold=0) should contain 50%
    # training data and 50% testing data.
    train_index, test_index = cv.get_split(X)
    assert len(train_index) == 3
    assert len(test_index) == 3
