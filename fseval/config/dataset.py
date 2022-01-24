from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING

from fseval.types import Task


@dataclass
class DatasetConfig:
    """
    Configures a dataset, to be used in the pipeline. Can be loaded from various sources
    using an 'adapter'.

    Attributes:
        name (str): Human-readable name of dataset.
        task (Task): Either Task.classification or Task.regression.
        adapter: Dataset adapter. must be of fseval.types.AbstractAdapter type,
            i.e. must implement a get_data() -> (X, y) method. Can also be a callable;
            then the callable must return a tuple (X, y).
        adapter_callable: Adapter class callable. the function to be called on the
            instantiated class to fetch the data (X, y). is ignored when the target
            itself is a function callable.
        feature_importances (Optional[Dict[str, float]]): Weightings indicating relevant
            features or instances. should be a dict with each key and value like the
            following pattern:
                X[<numpy selector>] = <float>
            Example:
                X[:, 0:3] = 1.0
            which sets the 0-3 features as maximally relevant and all others
            minimally relevant.
        group (Optional[str]): An optional group attribute, such to group datasets in
            the analytics stage.
        domain (Optional[str]): Dataset domain, e.g. medicine, finance, etc.
    """

    name: str = MISSING
    task: Task = MISSING
    adapter: Any = MISSING
    adapter_callable: str = "get_data"
    feature_importances: Optional[Dict[str, float]] = None
    # optional tags
    group: Optional[str] = None
    domain: Optional[str] = None
    # runtime properties: will be set once dataset is loaded, no need to configure them.
    n: Optional[int] = None
    p: Optional[int] = None
    multioutput: Optional[bool] = None

    # required for instantiation
    _target_: str = "fseval.pipeline.dataset.DatasetLoader"
    _recursive_: bool = False  # prevent adapter from getting initialized
