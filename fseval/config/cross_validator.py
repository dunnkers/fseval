from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class CrossValidatorConfig:
    """
    Provides an interface as how to define a Cross Validation method. The CV is applied
    at the beginning of the pipeline, so both the ranker and validator only get to see
    the **training** dataset. The test dataset is used for scoring, i.e. for determining
    the validation estimator scores.

    Attributes:
        name (str): Human-friendly name for this CV method.
        splitter (Any): The cross validation splitter function. Must be of type
            sklearn.model_selection.BaseCrossValidator or
            sklearn.model_selection.BaseShuffleSplit. In other words,
            the `splitter` argument should contain a _target_ attribute which
            instantiates to an object that has a `split` method with the following
            signature `def split(self, X, y=None, groups=None)`.
        fold (int): The fold to use in this specific run of the pipeline. e.g. you
            can use `python my_benchmark.py --multirun cv=kfold cv.splitter.n_spits=5 cv.fold=range(0,5)`
            to run a complete 5-fold CV scheme.
    """

    name: str = MISSING
    splitter: Any = None
    fold: int = 0

    # required for instantiation
    _target_: str = "fseval.pipeline.cv.CrossValidator"
