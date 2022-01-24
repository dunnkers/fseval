from dataclasses import dataclass
from typing import Any, List, Optional

from omegaconf import MISSING


@dataclass
class ResampleConfig:
    """
    Used to resample the dataset, before running the pipeline. Notice resampling is
    performed **after** cross validation. One can, for example, use resampling **with**
    replacement, such as to create multiple bootstraps of the dataset. In this way,
    algorithm stability can be approximated.

    Attributes:
        name (str): Human-friendly name of the resampling method.
        replace (bool): Whether to use resampling with replacement, yes or no.
        sample_size (Any): Can be one of two types: either a **float** from [0.0 to 1.0],
            such to select a **fraction** of the dataset to be sampled. Or,  an **int**
            from [1 to n_samples] can be used. This is the amount of exact samples to
            be selected.
        random_state (Optional[int]): Optionally, one might fix a random state to be
            used in the resampling process. In this way, results can be reproduced.
        stratify (Optional[List]): Whether to use stratified resampling. See
            sklearn.utils.resample for more information.
    """

    name: str = MISSING
    replace: bool = False
    sample_size: Any = None  # float [0.0 to 1.0] or int [1 to n_samples]
    random_state: Optional[int] = None
    stratify: Optional[List] = None

    # required for instantiation
    _target_: str = "fseval.pipeline.resample.Resample"
