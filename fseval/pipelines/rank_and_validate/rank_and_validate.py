from dataclasses import dataclass
from logging import Logger, getLogger
from typing import cast

import numpy as np
import pandas as pd
from fseval.callbacks import WandbCallback
from fseval.types import TerminalColor as tc
from fseval.utils.dict_utils import dict_flatten
from omegaconf import MISSING
from sklearn.base import clone
from tqdm import tqdm

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

    def score(self, X, y, **kwargs):
        scores = pd.DataFrame()

        # ranking scores
        ranking_score = self.ranking_validator.score(
            X, y, feature_importances=kwargs.get("feature_importances")
        )
        ranking_score["group"] = "ranking"
        scores = scores.append(ranking_score)

        # feature support scores - if available
        if self.ranker.estimates_feature_support:
            support_score = self.support_validator.score(X, y)
            support_score["group"] = "support"
            scores = scores.append(support_score)

        # validation scores
        validation_score = self.dataset_validator.score(X, y)
        validation_score["group"] = "validation"
        scores = scores.append(validation_score)

        # attach bootstrap and finish
        scores["bootstrap_state"] = self.bootstrap_state
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

    def score(self, X, y, **kwargs):
        scores = super(BootstrappedRankAndValidate, self).score(X, y, **kwargs)

        ranking_scores = scores[scores["group"] == "ranking"].dropna(axis=1)
        ranking_scores = ranking_scores.drop(columns=["group"])
        ranking_scores = ranking_scores.set_index("bootstrap_state")

        support_scores = scores[scores["group"] == "support"].dropna(axis=1)
        support_scores = support_scores.drop(columns=["group"])

        validation_scores = scores[scores["group"] == "validation"].dropna(axis=1)
        validation_scores = validation_scores.drop(columns=["group"])

        ##### Ranking scores - aggregation
        agg_ranking_scores = ranking_scores.agg(["mean", "std", "var", "min", "max"])
        # print scores
        self.logger.info(f"{tc.yellow(self.ranker.name)} ranking scores:")
        print(agg_ranking_scores)

        ##### Validation scores - aggregation
        val_scores_per_feature = validation_scores.groupby("n_features_to_select")
        # print scores
        print()
        self.logger.info(f"{tc.yellow(self.validator.name)} validation scores:")
        agg_val_scores = val_scores_per_feature.mean().drop(columns=["bootstrap_state"])
        print(agg_val_scores)

        ##### Summary
        summary = dict()
        ### Mean ranking score
        ranking_scores_mean = ranking_scores.agg(["mean"])
        ranking_scores_mean = ranking_scores_mean.add_prefix("ranker/")
        ranking_scores_mean_dict = ranking_scores_mean.to_dict()
        ranking_scores_mean_flat = dict_flatten(ranking_scores_mean_dict, sep="/")
        summary = dict(**summary, **ranking_scores_mean_flat)
        ### Mean validation score
        validation_scores_mean = agg_val_scores.agg(["mean"])
        validation_scores_mean = validation_scores_mean.add_prefix("validator/")
        validation_scores_mean_dict = validation_scores_mean.to_dict()
        validation_scores_mean_flat = dict_flatten(validation_scores_mean_dict, sep="/")
        summary = dict(**summary, **validation_scores_mean_flat)

        ##### Upload tables
        wandb_callback = getattr(self.callbacks, "wandb", False)
        if wandb_callback:
            self.logger.info(f"Uploading tables to wandb...")
            wandb_callback = cast(WandbCallback, wandb_callback)

        ### ranking scores
        if wandb_callback and self.upload_ranking_scores:
            self.logger.info(f"Uploading ranking scores...")

            ## upload raw rankings
            # feature importances
            if self.ranker.estimates_feature_importances:
                importances_table = self._get_ranker_attribute_table(
                    "feature_importances"
                )
                wandb_callback.upload_table(importances_table, "feature_importances")

            # feature support
            if self.ranker.estimates_feature_support:
                support_table = self._get_ranker_attribute_table("feature_support")
                wandb_callback.upload_table(support_table, "feature_support")

            # feature ranking
            if self.ranker.estimates_feature_ranking:
                ranking_table = self._get_ranker_attribute_table("feature_ranking")
                wandb_callback.upload_table(ranking_table, "feature_ranking")

            ## upload ranking scores
            wandb_callback.upload_table(ranking_scores.reset_index(), "ranking_scores")

        ### validation scores
        if wandb_callback and self.upload_validation_scores:
            self.logger.info(f"Uploading validation scores...")

            ## upload support scores
            if self.ranker.estimates_feature_support:
                wandb_callback.upload_table(support_scores, "support_scores")

            ## upload validation scores
            wandb_callback.upload_table(validation_scores, "validation_scores")

            ## upload mean validation scores
            all_agg_val_scores = agg_val_scores.reset_index()
            wandb_callback.upload_table(all_agg_val_scores, "validation_scores_mean")

        if wandb_callback:
            self.logger.info(f"Tables uploaded {tc.green('✓')}")

        ##### Upload charts
        # has ground truth
        rank_and_validate_estimator = self.estimators[0]
        ranking_validator_estimator = rank_and_validate_estimator.ranking_validator
        X_importances = ranking_validator_estimator.X_importances
        has_ground_truth = X_importances is not None

        ### Aggregated charts
        if wandb_callback:
            # mean validation scores
            wandb_callback.add_panel(
                panel_name="mean_validation_score",
                viz_id="dunnkers/fseval/datasets-vs-rankers",
                table_key="validation_scores_mean",
                config_fields=[
                    "ranker/name",
                    "dataset/name",
                    "validator/name",
                    "dataset/task",
                ],
                fields={
                    "validator": "validator/name",
                    "score": "score",
                    "x": "ranker/name",
                    "y": "dataset/name",
                    "task": "dataset/task",
                },
                string_fields={
                    "title": "Feature ranker performance",
                    "subtitle": "→ Mean validation score over all bootstraps",
                    "ylabel": "accuracy or r2-score",
                    "aggregation_op": "mean",
                    "scale_type": "pow",
                    "color_exponent": "5",
                    "opacity_exponent": "15",
                    "scale_string": "x^5",
                    "color_scheme": "redyellowblue",
                    "reverse_colorscheme": "",
                    "text_threshold": "0.95",
                },
            )

            # stability scores
            wandb_callback.add_panel(
                panel_name="feature_importance_stability",
                viz_id="dunnkers/fseval/datasets-vs-rankers",
                table_key="feature_importances",
                config_fields=[
                    "ranker/name",
                    "dataset/name",
                    "validator/name",
                    "dataset/task",
                ],
                fields={
                    "validator": "validator/name",
                    "score": "feature_importances",
                    "x": "ranker/name",
                    "y": "dataset/name",
                    "task": "dataset/task",
                },
                string_fields={
                    "title": "Algorithm Stability",
                    "subtitle": "→ Mean stdev of feature importances. Lower is better.",
                    "aggregation_op": "stdev",
                    "scale_type": "quantile",
                    "color_exponent": "5",
                    "opacity_exponent": "0",
                    "scale_string": "qnt",
                    "color_scheme": "reds",
                    "reverse_colorscheme": "",
                    "text_threshold": "0.95",
                },
            )

            # fitting time
            wandb_callback.add_panel(
                panel_name="fitting_times",
                viz_id="dunnkers/fseval/datasets-vs-rankers",
                table_key="ranking_scores",
                config_fields=[
                    "ranker/name",
                    "dataset/name",
                    "validator/name",
                    "dataset/task",
                ],
                fields={
                    "validator": "validator/name",
                    "score": "fit_time",
                    "x": "ranker/name",
                    "y": "dataset/name",
                    "task": "dataset/task",
                },
                string_fields={
                    "title": "Fitting time (seconds)",
                    "subtitle": "→ As mean over all bootstraps",
                    "aggregation_op": "mean",
                    "scale_type": "pow",
                    "color_exponent": "0.1",
                    "opacity_exponent": "15",
                    "scale_string": "x^0.1",
                    "color_scheme": "redyellowblue",
                    "reverse_colorscheme": "true",
                    "text_threshold": "0.95",
                },
            )

        ### Individual charts
        if wandb_callback:
            # validation scores bootstraps
            wandb_callback.add_panel(
                panel_name="validation_score_bootstraps",
                viz_id="dunnkers/fseval/validation-score-bootstraps",
                table_key="validation_scores",
                config_fields=["validator/name", "ranker/name"],
                string_fields={
                    "title": "Classification accuracy vs. Subset size",
                    "subtitle": "→ for all bootstraps",
                    "ylabel": "accuracy or r2-score",
                },
            )

            # validation scores
            wandb_callback.add_panel(
                panel_name="validation_score",
                viz_id="dunnkers/fseval/validation-score",
                table_key="validation_scores",
                config_fields=["ranker/name"],
                fields={
                    "hue": "ranker/name",
                },
                string_fields={
                    "title": "Classification accuracy vs. Subset size",
                    "subtitle": "→ as the mean over all bootstraps",
                    "ylabel": "accuracy or r2-score",
                },
            )

            # mean feature importance
            wandb_callback.add_panel(
                panel_name="feature_importances_mean",
                viz_id="wandb/bar/v0",
                table_key="feature_importances",
                fields={
                    "label": "feature_index",
                    "value": "feature_importances",
                },
                string_fields={"title": "Feature importance per feature"},
            )

            # feature importance & stability
            wandb_callback.add_panel(
                panel_name="feature_importances_stability",
                viz_id="dunnkers/fseval/feature-importances-stability",
                table_key="feature_importances",
                fields={
                    "x": "feature_index",
                    "y": "feature_importances",
                },
                string_fields={
                    "title": "Feature importance & Stability",
                    "subtitle": "→ a smaller stdev means more stability",
                },
            )

            # feature importance vs feature index
            wandb_callback.add_panel(
                panel_name="feature_importances_all_bootstraps",
                viz_id="dunnkers/fseval/feature-importances-all-bootstraps-with-ticks",
                table_key="feature_importances",
                fields={
                    "x": "feature_index",
                    "y": "feature_importances",
                },
                string_fields={
                    "title": "Feature importance vs. Feature index",
                    "subtitle": "→ estimated feature importance per feature",
                },
            )

        return summary
