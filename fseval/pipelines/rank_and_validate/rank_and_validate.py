from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, Union, cast

import numpy as np
import pandas as pd
from fseval.callbacks import WandbCallback
from fseval.types import TerminalColor as tc
from fseval.utils.dict_utils import dict_flatten
from omegaconf import MISSING
from sklearn.base import clone

from .._experiment import Experiment
from ._config import RankAndValidatePipeline
from ._dataset_validator import DatasetValidator
from ._ranking_validator import RankingValidator
from ._support_validator import SupportValidator


@dataclass
class RankAndValidate(Experiment, RankAndValidatePipeline):
    """First fits a feature ranker (or feature selector, as long as the estimator will
    attach at `feature_importances_` property to its class), and then validates the
    feature ranking by fitting a 'validation' estimator. The validation estimator can be
    any normal sklearn estimator; just as long it supports the specified dataset type -
    regression or classification."""

    bootstrap_state: int = MISSING
    logger: Logger = getLogger(__name__)

    def _get_estimator(self):
        estimators = []

        # make sure to clone the validator for support- and dataset validation.
        config = self._get_config()
        validator = config.pop("validator")

        ## first fit ranker, then run all validations
        # instantiate ranking validator. pass bootstrap state for the cache filename
        self.ranking_validator = RankingValidator(
            **self._get_config(), bootstrap_state=self.bootstrap_state
        )
        estimators.append(self.ranking_validator)

        # validate feature support - if available
        if self.ranker.estimates_feature_support:
            self.support_validator = SupportValidator(
                **config,
                validator=clone(validator),
                bootstrap_state=self.bootstrap_state,
            )
            estimators.append(self.support_validator)

        # instantiate dataset validator.
        self.dataset_validator = DatasetValidator(
            **config, validator=clone(validator), bootstrap_state=self.bootstrap_state
        )
        estimators.append(self.dataset_validator)

        return estimators

    def _prepare_data(self, X, y):
        # resample dataset: perform a bootstrap
        self.resample.random_state = self.bootstrap_state
        X, y = self.resample.transform(X, y)

        return X, y

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        scores = {}

        # ranking scores
        ranking_scores = self.ranking_validator.score(
            X, y, feature_importances=kwargs.get("feature_importances")
        )
        ranking_scores["bootstrap_state"] = self.bootstrap_state
        scores["ranking"] = ranking_scores

        # feature support scores - if available
        if self.ranker.estimates_feature_support:
            support_scores = self.support_validator.score(X, y)
            support_scores["bootstrap_state"] = self.bootstrap_state
            scores["support"] = support_scores

        # validation scores
        validation_scores = self.dataset_validator.score(X, y)
        validation_scores["bootstrap_state"] = self.bootstrap_state
        scores["validation"] = validation_scores

        # custom metrics
        for metric_name, metric_class in self.metrics.items():
            scores_metric = metric_class.score_pipeline(scores, self.callbacks)

            if scores_metric is not None:
                scores = scores_metric

        # finish
        self.logger.info(
            f"scored bootstrap_state={self.bootstrap_state} " + tc.green("✓")
        )

        return scores


@dataclass
class BootstrappedRankAndValidate(Experiment, RankAndValidatePipeline):
    """Provides an experiment that performs a 'bootstrap' procedure: using different
    `random_state` seeds the dataset is continuously resampled with replacement, such
    that various metrics can be better approximated."""

    logger: Logger = getLogger(__name__)

    def _get_n_jobs(self):
        """Allow each bootstrap experiment to run on a separate CPU."""

        return self.n_jobs

    def _get_estimator(self):
        for bootstrap_state in np.arange(1, self.n_bootstraps + 1):
            config = self._get_config()
            ranker = config.pop("ranker")

            yield RankAndValidate(
                **config,
                ranker=clone(ranker),
                bootstrap_state=bootstrap_state,
            )

    def _get_overrides_text(self, estimator):
        return f"[bootstrap_state={estimator.bootstrap_state}] "

    def _get_ranker_attribute_table(self, attribute: str):
        # ensure dataset loaded
        p = self.dataset.p
        assert p is not None, "dataset must be loaded"

        # construct attribute table
        attribute_table = pd.DataFrame()

        def get_attribute_row(attribute_value: np.ndarray, group: str):
            attribute_data = {
                "feature_index": np.arange(1, p + 1),  # type: ignore
            }
            attribute_data[attribute] = attribute_value
            attribute_data["group"] = group
            attribute_row = pd.DataFrame(attribute_data)

            return attribute_row

        # attach estimated attributes
        has_ground_truth = False
        for rank_and_validate in self.estimators:
            # get attributes from rank and validate estimator
            ranking_validator = rank_and_validate.ranking_validator
            bootstrap_state = ranking_validator.bootstrap_state
            has_ground_truth = ranking_validator.X_importances is not None

            # attach estimated
            estimated = getattr(ranking_validator, f"estimated_{attribute}")
            estimated_row = get_attribute_row(estimated, "estimated")
            estimated_row["bootstrap_state"] = bootstrap_state
            attribute_table = attribute_table.append(estimated_row)

        # attach ground truth, if available
        if has_ground_truth:
            ground_truth = getattr(ranking_validator, f"ground_truth_{attribute}")
            ground_truth_row = get_attribute_row(ground_truth, "ground_truth")
            attribute_table = attribute_table.append(ground_truth_row)

        return attribute_table

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        scores = super(BootstrappedRankAndValidate, self).score(X, y, **kwargs)
        assert isinstance(
            scores, Dict
        ), "Scores returned from `rank_and_validate` (the pipeline) must be dict's."
        scores = cast(Dict, scores)

        # custom metrics
        for metric_name, metric_class in self.metrics.items():
            scores_metric = metric_class.score_bootstrap(
                self.ranker, self.validator, self.callbacks, scores
            )

            if scores_metric is not None:
                scores = scores_metric

        return {}
