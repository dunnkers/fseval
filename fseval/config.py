from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from fseval.types import CacheUsage, Task


@dataclass
class CrossValidatorConfig:
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = "fseval.pipeline.cv.CrossValidator"
    name: str = MISSING
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` function. """
    splitter: Any = None
    fold: int = 0


@dataclass
class DatasetConfig:
    """
    Args:
        name: human-readable name of dataset.

        group: an optional group attribute, such to group datasets in the analytics
        stage.

        task: either Task.classification or Task.regression.

        domain: dataset domain, e.g. medicine, finance, etc.

        adapter: dataset adapter. must be of fseval.types.AbstractAdapter type,
        i.e. must implement a get_data() -> (X, y) method.

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

    _target_: str = "fseval.pipeline.dataset.DatasetLoader"
    _recursive_: bool = False  # prevent adapter from getting initialized
    name: str = MISSING
    task: Task = MISSING
    adapter: Any = MISSING
    adapter_callable: str = "get_data"
    feature_importances: Optional[Dict[str, float]] = None
    # optional tags
    group: Optional[str] = None
    domain: Optional[str] = None
    # runtime properties: set once dataset is loaded
    n: Optional[int] = None
    p: Optional[int] = None
    multioutput: Optional[bool] = None


@dataclass
class EstimatorConfig:
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    load_cache: Optional[CacheUsage] = None
    save_cache: Optional[CacheUsage] = None
    # tags
    multioutput: Optional[bool] = None
    multioutput_only: Optional[bool] = None
    requires_positive_X: Optional[bool] = None
    estimates_feature_importances: Optional[bool] = None
    estimates_feature_support: Optional[bool] = None
    estimates_feature_ranking: Optional[bool] = None
    estimates_target: Optional[bool] = None


@dataclass
class TaskedEstimatorConfig(EstimatorConfig):
    _target_: str = "fseval.pipeline.estimator.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    _target_class_: str = "fseval.pipeline.estimator.Estimator"
    name: str = MISSING
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None
    load_cache: CacheUsage = CacheUsage.allow
    save_cache: CacheUsage = CacheUsage.allow
    # tags
    multioutput: bool = False
    multioutput_only: bool = False
    requires_positive_X: bool = False
    estimates_feature_importances: bool = False
    estimates_feature_support: bool = False
    estimates_feature_ranking: bool = False
    estimates_target: bool = False
    # runtime properties
    task: Task = II("dataset.task")
    is_multioutput_dataset: bool = II("dataset.multioutput")


@dataclass
class ResampleConfig:
    _target_: str = "fseval.pipeline.resample.Resample"
    name: str = MISSING
    replace: bool = False
    sample_size: Any = None  # float [0.0 to 1.0] or int [1 to n_samples]
    random_state: Optional[int] = None
    stratify: Optional[List] = None


@dataclass
class StorageConfig:
    _target_: str = MISSING
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None


@dataclass
class PipelineConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": MISSING},
            {"ranker": MISSING},
            {"validator": MISSING},
            {"cv": "kfold"},
            {"storage": "local"},
            {"resample": "shuffle"},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
    pipeline: str = "rank-and-validate"
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(_target_="fseval.storage.MockStorage")
    )
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    callbacks: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "fseval.pipelines._callback_collection.CallbackCollection"
        }
    )
    n_bootstraps: int = 1
    n_jobs: Optional[int] = 1
    all_features_to_select: str = "range(1, min(50, p) + 1)"
    metrics: Dict[str, Any] = field(default_factory=dict)
