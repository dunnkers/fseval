import logging
from dataclasses import dataclass
from typing import Dict, Optional

from omegaconf import MISSING

from fseval.pipeline.estimator import Estimator
from fseval.pipeline.resample import Resample
from fseval.types import AbstractMetric

from .._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class RankAndValidatePipeline(Pipeline):
    """Instantiated version of `PipelineConfig`: the actual pipeline
    implementation."""

    pipeline: str = MISSING
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    n_bootstraps: int = MISSING
    n_jobs: Optional[int] = MISSING
    all_features_to_select: str = MISSING
    metrics: Dict[str, AbstractMetric] = MISSING

    def _get_config(self):
        return {
            key: getattr(self, key)
            for key in RankAndValidatePipeline.__dataclass_fields__.keys()
        }
