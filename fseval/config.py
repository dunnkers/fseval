from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


class Task(Enum):
    regression = 1
    classification = 2


@dataclass
class GroupItem:
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
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    multivariate: bool = False


@dataclass
class TaskedEstimatorConfig(GroupItem):
    _target_: str = "fseval.base.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    task: Task = MISSING
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None
    # instance_based_ranking: bool = False


@dataclass
class CallbacksConfig:
    stdout: bool = False
    wandb: bool = False


@dataclass
class PipelineConfig:
    _target_: str = MISSING  # will be main class of entire program
    # callbacks: List[CallbackConfig] = field(default_factory=lambda: [])


# FEATURE_RANKING_DEFAULTS = [
#     {"/estimator@ranker": "chi2"},
# ]


@dataclass
class FeatureRankingConfig(PipelineConfig):
    # defaults: List[Any] = field(default_factory=lambda: FEATURE_RANKING_DEFAULTS)

    _target_: str = "fseval.pipeline.FeatureRanking"
    job_type: str = "feature-ranking"
    # in CLI, use: estimator@pipeline.ranker=chi2
    ranker: TaskedEstimatorConfig = MISSING
    # task: Task = II("dataset.task")
    # adapter: Any = None


RUN_ESTIMATOR_DEFAULTS = [
    {"/estimator": "dt"},
]


@dataclass
class RunEstimatorConfig:
    _target_: str = "fseval.pipeline.RunEstimator"
    job_type: str = "run-estimator"
    # in CLI, use: estimator@pipeline.ranker=chi2
    validator: TaskedEstimatorConfig = MISSING


# RANK_AND_VALIDATE_DEFAULTS = FEATURE_RANKING_DEFAULTS + RUN_ESTIMATOR_DEFAULTS


# @dataclass
# class RankAndValidateConfig(FeatureRankingConfig, RunEstimatorConfig):
#     defaults: List[Any] = field(default_factory=lambda: RANK_AND_VALIDATE_DEFAULTS)


@dataclass
class BaseConfig:
    _target_: str = MISSING
    # these are passed into pipeline class
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    wandb: Dict = field(default_factory=lambda: dict())
    callbacks: Dict = field(default_factory=lambda: dict())
    # pipeline; is instantiated with all above objects
    pipeline: Any = MISSING


cs = ConfigStore.instance()
cs.store(group="pipeline", name="base_feature_ranking", node=FeatureRankingConfig)
cs.store(group="pipeline", name="base_run_estimator", node=RunEstimatorConfig)
# cs.store(group="pipeline", name="rank_and_validate", node=RankAndValidateConfig)
# cs.store(group="task", name="")
# cs.store(name="estimator", node=EstimatorConfig)
cs.store(name="base_config", node=BaseConfig)
