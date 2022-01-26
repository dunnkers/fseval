from omegaconf import MISSING

from fseval.config import PipelineConfig

# To set PipelineConfig defaults in a Structured Config, we must redefine the entire
# defaults list.
my_config = PipelineConfig(
    n_bootstraps=1,
    defaults=[
        "_self_",
        # highlight-next-line
        {"dataset": "synthetic"},
        {"cv": "kfold"},
        {"resample": "shuffle"},
        {"ranker": MISSING},
        # highlight-next-line
        {"validator": "knn"},
        {"storage": "local"},
        # highlight-next-line
        {"callbacks": ["to_sql"]},
        {"metrics": ["feature_importances", "ranking_scores", "validation_scores"]},
        {"override hydra/job_logging": "colorlog"},
        {"override hydra/hydra_logging": "colorlog"},
    ],
)
