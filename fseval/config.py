from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from fseval.types import CacheUsage, Task

cs = ConfigStore.instance()


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


@dataclass
class EstimatorConfig:
    """
    Configures an estimator: a Feature Ranker, Feature Selector or a validation
    estimator. In the case of Feature Rankers/Selectors, set one of
    `estimates_feature_importances`, `estimates_feature_support` or
    `estimates_feature_ranking` to True. In the case of a validation estimator,
    set `estimates_target` to True.

    Attributes:
        name (str): Human-friendly name of this estimator.
        estimator (Any): The estimator. Must be a dictionary with a key `_target_`,
            pointing to the class that is to be instantiated. All other properties in
            the dictionary will be passed to the estimator constructor. e.g.:
            `_target_="sklearn.tree.DecisionTreeClassifier, max_depth=10)`
        load_cache (Optional[CacheUsage]): How to handle loading a cached version of the
            estimator, in a pickle file. e.g. to ignore cache, or force using it.
            To be used in combination with `PipelineConfig.storage`. See `CacheUsage`.
        save_cache (Optional[CacheUsage]): How to handle saving the fit estimator as a
            pickle file, such to facilitate caching. To be used in combination with
            `PipelineConfig.storage`. See `CacheUsage`.
        _estimator_type (str): Either 'classifier', 'regressor' or 'clusterer'.
        multioutput (bool): Whether this estimator supports multioutput datasets.
        multioutput_only (bool): If this estimator **only** supports multioutput
            datasets.
        requires_positive_X (bool): Whether the estimator fails if X contains negative
            values.
        estimates_feature_importances (bool): Whether the estimator estimates feature
            importances. For example, in the case of 2 features, the estimator can set
            `self.feature_importances_ = [0.9, 0.1]`, implying the estimator
            found the first feature the most useful. Alternatively, the `coef_`
            attribute can also be read and interpreted as a feature importance vector.
        estimates_feature_support (bool): Whether the estimator estimates feature
            support. A feature support vector indicates which features to include in a
            feature subset yes/no. In other words, it must be a boolean vector. It is
            to be set on the estimator `support_` attribute. Estimating the feature
            support vector is synonymous with performing feature selection. e.g.:
            `self.support_ = [True, False]`, meaning to include only the first feature
            in a feature subset.
        estimates_feature_ranking (bool): Whether the estimator ranks the features in a
            specific order. Is similar to feature importance, but does not estimate
            exact importance quantities, i.e. that are proportional to each other. An
            estimator can set the ranking using the `ranking_` attribute. e.g.:
            `self.ranking_ = [1, 0]`, such to indicate that the first feature ranks the
            highest.
    """

    name: str = MISSING
    estimator: Any = None
    load_cache: Optional[CacheUsage] = CacheUsage.allow
    save_cache: Optional[CacheUsage] = CacheUsage.allow
    # tags
    _estimator_type: str = MISSING  # 'classifier', 'regressor' or 'clusterer'
    multioutput: bool = False
    multioutput_only: bool = False
    requires_positive_X: bool = False
    estimates_feature_importances: bool = False
    estimates_feature_support: bool = False
    estimates_feature_ranking: bool = False
    estimates_target: bool = False
    # runtime properties. do not override these.
    task: Task = II("dataset.task")
    is_multioutput_dataset: bool = II("dataset.multioutput")

    # required for instantiation
    _target_: str = "fseval.pipeline.estimator.Estimator"


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
    """
    The complete configuration needed to run the fseval pipeline.

    Attributes:
        dataset (DatasetConfig): Determines the dataset to use for this experiment.
        ranker (EstimatorConfig): A Feature Ranker or Feature Selector.
        validator (EstimatorConfig): Some estimator to validate the feature subsets.
        cv (CrossValidatorConfig): The CV method and split to use in this experiment.
        resample (ResampleConfig): Dataset resampling; e.g. with or without replacement.
        storage (StorageConfig): A storage method used to store the fit estimators.
        callbacks (Dict[str, Any]): Callbacks. Provide hooks for storing the config or
            results.
        metrics (Dict[str, Any]): Metrics allow custom computation after any pipeline
            stage.
        n_bootstraps (int): Amount of 'bootstraps' to run. A bootstrap means running
            the pipeline again but with a resampled (see `resample`) version of the
            dataset. This allows estimating stability, for example.
        n_jobs (Optional[int]): Amount of CPU's to use for computing each bootstrap.
            This thus distributes the amount of bootstraps over CPU's.
        all_features_to_select (str): Once the ranker has been fit, this determines the
            feature subsets to validate. By default, at most 50 subsets containing the
            highest ranked features are validated.
    """

    dataset: DatasetConfig = MISSING
    ranker: EstimatorConfig = MISSING
    validator: EstimatorConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    storage: StorageConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=lambda: {})
    metrics: Dict[str, Any] = field(default_factory=lambda: {})
    n_bootstraps: int = 1
    n_jobs: Optional[int] = 1
    all_features_to_select: str = "range(1, min(50, p) + 1)"

    # default values for the above.
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": MISSING},
            {"ranker": MISSING},
            {"validator": MISSING},
            {"cv": "kfold"},
            {"storage": "local"},
            {"resample": "shuffle"},
            {"callbacks": []},
            {"metrics": ["feature_importances", "ranking_scores", "validation_scores"]},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )

    # required for instantiation
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"


cs.store("base_pipeline_config", PipelineConfig)
