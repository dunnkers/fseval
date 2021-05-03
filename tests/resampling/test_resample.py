from fseval.resampling import Resample
import numpy as np


def test_resample_shuffling():
    resampler = Resample(random_state=0)
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    X_shuffled = resampler.transform(X)
    assert not (np.array(X) == np.array(X_shuffled)).all()
