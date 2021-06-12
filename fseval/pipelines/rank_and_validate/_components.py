from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional, cast

import numpy as np
import pandas as pd
from fseval.callbacks import WandbCallback
from fseval.types import TerminalColor as tc
from omegaconf import MISSING
from sklearn.base import clone
from tqdm import tqdm

from .._experiment import Experiment
from ._config import RankAndValidatePipeline
from ._dataset_validator import DatasetValidator
from ._ranking_validator import RankingValidator


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
        # instantiate ranking validator. pass bootstrap state for the cache filename
        self.ranking_validator = RankingValidator(
            **self._get_config(), bootstrap_state=self.bootstrap_state
        )

        # instantiate dataset validator.
        self.dataset_validator = DatasetValidator(
            **self._get_config(), bootstrap_state=self.bootstrap_state
        )

        # first fit ranker, then run all validations
        return [
            self.ranking_validator,
            self.dataset_validator,
        ]

    def _prepare_data(self, X, y):
        # resample dataset: perform a bootstrap
        self.resample.random_state = self.bootstrap_state
        X, y = self.resample.transform(X, y)
        return X, y

    def score(self, X, y, **kwargs):
        ranking_score = self.ranking_validator.score(
            X, y, feature_importances=kwargs.get("feature_importances")
        )
        ranking_score["group"] = "ranking"

        validation_score = self.dataset_validator.score(X, y)
        validation_score["group"] = "validation"

        scores = pd.DataFrame()
        scores = scores.append(ranking_score)
        scores = scores.append(validation_score)
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

    def _get_ranker_attribute_table(self, attribute: str, attribute_name: str):
        attribute_table = pd.DataFrame()

        for rank_and_validate in self.estimators:
            ranker = rank_and_validate.ranker
            attribute_value = getattr(ranker, attribute)
            p = self.dataset.p
            assert p is not None, "dataset must be loaded"
            attribute_data = {
                "bootstrap_state": rank_and_validate.bootstrap_state,
                "feature_index": np.arange(1, p + 1),
            }
            attribute_data[attribute_name] = attribute_value
            attribute_row = pd.DataFrame(attribute_data)
            attribute_table = attribute_table.append(attribute_row)

        return attribute_table

    def score(self, X, y, **kwargs):
        scores = super(BootstrappedRankAndValidate, self).score(X, y, **kwargs)

        ranking_scores = scores[scores["group"] == "ranking"].dropna(axis=1)
        ranking_scores = ranking_scores.drop(columns=["group"])
        ranking_scores = ranking_scores.set_index("bootstrap_state")
        validation_scores = scores[scores["group"] == "validation"].dropna(axis=1)
        validation_scores = validation_scores.drop(columns=["group"])

        ##### Ranking scores - aggregation
        agg_ranking_scores = ranking_scores.agg(["mean", "std", "var", "min", "max"])
        # print scores
        self.logger.info(f"{tc.yellow(self.ranker.name)} ranking scores:")
        print(agg_ranking_scores)
        # send metrics
        agg_ranking_scores = agg_ranking_scores.to_dict()
        self.callbacks.on_metrics(dict(ranker=agg_ranking_scores))

        ##### Validation scores - aggregation
        val_scores_per_feature = validation_scores.groupby("n_features_to_select")
        progbar = tqdm(list(val_scores_per_feature), desc="uploading validation scores")

        for n_features_to_select, feature_scores in progbar:
            feature_scores = feature_scores.drop(
                columns=["bootstrap_state", "n_features_to_select"]
            )
            agg_feature_scores = feature_scores.agg(
                ["mean", "std", "var", "min", "max"]
            )

            # send metrics
            agg_feature_scores_dict = agg_feature_scores.to_dict()
            agg_feature_scores_dict["n_features_to_select"] = int(n_features_to_select)
            self.callbacks.on_metrics(dict(validator=agg_feature_scores_dict))
        # print scores
        print()
        self.logger.info(f"{tc.yellow(self.validator.name)} validation scores:")
        agg_val_scores = val_scores_per_feature.mean().drop(columns=["bootstrap_state"])
        print(agg_val_scores)

        ##### Summary
        best = {}
        # best validator
        best_subset_index = validation_scores["score"].argmax()
        best_subset = validation_scores.iloc[best_subset_index]
        best["validator"] = best_subset.to_dict()
        # best accompanying ranking
        best_subset_bootstrap_state = best_subset["bootstrap_state"]
        best_ranker = ranking_scores.loc[best_subset_bootstrap_state]
        best["ranker"] = best_ranker.to_dict()
        # summary
        summary = dict(best=best)

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
                    "feature_importances_", "feature_importances"
                )
                wandb_callback.upload_table(importances_table, "feature_importances")

            # feature support
            if self.ranker.estimates_feature_support:
                support_table = self._get_ranker_attribute_table(
                    "feature_support_", "feature_support"
                )
                wandb_callback.upload_table(support_table, "feature_support")

            # feature ranking
            if self.ranker.estimates_feature_ranking:
                ranking_table = self._get_ranker_attribute_table(
                    "feature_ranking_", "feature_ranking"
                )
                wandb_callback.upload_table(ranking_table, "feature_ranking")

            ## upload ranking scores
            wandb_callback.upload_table(ranking_scores.reset_index(), "ranking_scores")

        ### validation scores
        if wandb_callback and self.upload_validation_scores:
            self.logger.info(f"Uploading validation scores...")

            ## upload validation scores
            wandb_callback.upload_table(validation_scores, "validation_scores")

            ## upload mean validation scores
            all_agg_val_scores = agg_val_scores.reset_index()
            wandb_callback.upload_table(all_agg_val_scores, "validation_scores_mean")

        ### upload best scores
        if wandb_callback and self.upload_best_scores:
            self.logger.info(f"Uploading best scores...")

            ## best ranker- and validation scores
            best_subset_prefixed = best_subset.add_prefix("validator.")
            best_ranker_prefixed = best_ranker.add_prefix("ranker.")
            best_scores = pd.concat([best_subset_prefixed, best_ranker_prefixed])
            best_scores_df = pd.DataFrame([best_scores])
            wandb_callback.upload_table(best_scores_df, "best_scores")

        if wandb_callback:
            self.logger.info(f"Tables uploaded {tc.green('✓')}")

        return summary
