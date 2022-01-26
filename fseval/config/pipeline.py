from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .cross_validator import CrossValidatorConfig
from .dataset import DatasetConfig
from .estimator import EstimatorConfig
from .resample import ResampleConfig
from .storage import StorageConfig

cs = ConfigStore.instance()


@dataclass
class PipelineConfig:
    """
    The complete configuration needed to run the fseval pipeline.

    Attributes:
        dataset (DatasetConfig): Determines the dataset to use for this experiment.
        cv (CrossValidatorConfig): The CV method and split to use in this experiment.
        resample (ResampleConfig): Dataset resampling; e.g. with or without replacement.
        ranker (EstimatorConfig): A Feature Ranker or Feature Selector.
        validator (EstimatorConfig): Some estimator to validate the feature subsets.
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
        all_features_to_select (str): Once the ranker has been fit, this determines
            the feature subsets to validate. By default, at most 50 subsets containing
            the highest ranked features are validated. The format of this parameter is
            a string that can contain an arbitrary Python expression. The condition is
            that the expression must evaluate to a `List[int]` object. For example, the
            default is: `range(1, min(50, p) + 1)`. Each number in the list is passed
            to the `sklearn.feature_selection.SelectFromModel` as the `max_features`
            parameter. To see how the expression is evaluated, check out the
            `fseval.pipelines.rank_and_validate._dataset_validator` module.
        defaults (List[Any]): Default values for the above.
    """

    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    ranker: EstimatorConfig = MISSING
    validator: EstimatorConfig = MISSING
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
            {"cv": "kfold"},
            {"resample": "shuffle"},
            {"ranker": MISSING},
            {"validator": MISSING},
            {"storage": "local"},
            {"callbacks": []},
            {"metrics": ["feature_importances", "ranking_scores", "validation_scores"]},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )

    # required for instantiation
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"


cs.store("base_pipeline_config", PipelineConfig)
