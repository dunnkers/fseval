from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Task(Enum):
    regression = 1
    classification = 2


@dataclass
class GroupItem:
    _target_: str = MISSING
    name: str = MISSING


@dataclass
class DatasetConfig(GroupItem):
    """
    Args:
        name: human-readable name of dataset.

        task: either Task.classification or Task.regression.

        adapter: dataset adapter. must be of fseval.adapters.Adapter type, i.e. must
        implement a get_data() -> (X, y) method.

        adapter_callable: adapter class callable. the function to be called on the
        instantiated class to fetch the data (X, y). is ignored when the target itself
        is a function callable.

        feature_importances: weightings indicating relevant features or instances.
        should be a dict with each key and value like the following pattern:
            X[<numpy selector>] = <float>
        Example:
            X[:, 0:3] = 1.0
        which sets the 0-3 features as maximally relevant and all others
        minimally relevant.
    """

    _target_: str = "fseval.datasets.Dataset"
    _recursive_: bool = False  # prevent adapter from getting initialized
    task: Task = MISSING
    adapter: Any = MISSING
    adapter_callable: str = "get_data"
    feature_importances: Optional[Dict[str, float]] = None


@dataclass
class CrossValidatorConfig(GroupItem):
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = "fseval.cv.CrossValidator"
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` function. """
    splitter: Any = None
    fold: int = 0


@dataclass
class ResampleConfig(GroupItem):
    _target_: str = "fseval.resampling.Resample"
    replace: bool = False
    sample_size: Any = None  # float [0.0 to 1.0] or int [1 to n_samples]
    random_state: Optional[int] = None
    stratify: Optional[List] = None


@dataclass
class EstimatorConfig(GroupItem):
    _target_: str = "fseval.base.ConfigurableEstimator"
    task: Task = MISSING
    """ classifier. must have _target_ of BaseEstimator type with fit() method. """
    classifier: Any = None
    multivariate_clf: bool = False
    """ regressor. must have _target_ of BaseEstimator type with fit() method. """
    regressor: Any = None
    multivariate_reg: bool = False


@dataclass
class RankerConfig(EstimatorConfig):
    _target_: str = "fseval.rankers.Ranker"
    instance_based: bool = False


@dataclass
class ValidatorConfig(EstimatorConfig):
    _target_: str = "fseval.validators.Validator"


@dataclass
class ExperimentConfig:
    _target_: str = "fseval.experiment.Experiment"
    pipeline: Any = MISSING
    # pipeline specific
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    # wandb configuration semantics
    project: str = MISSING
    group: Optional[str] = None
    id: Optional[str] = None


@dataclass
class PipelineConfig:
    target_class: str = MISSING


@dataclass
class RunEstimatorConfig:
    estimator: EstimatorConfig = MISSING


@dataclass
class FeatureRankingConfig(PipelineConfig):
    target_class: str = "fseval.pipeline.FeatureRanking"


@dataclass
class BaseConfig:
    _target_: str = MISSING
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    estimator: EstimatorConfig = MISSING

    # pipeline
    pipeline: Any = MISSING
    wandb: Dict = field(default_factory=lambda: dict())


cs = ConfigStore.instance()
cs.store(group="pipeline", name="feature_ranking", node=FeatureRankingConfig)
cs.store(name="estimator", node=EstimatorConfig)
cs.store(name="base_config", node=BaseConfig)
