from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List, cast

import numpy as np
import pandas as pd
from fseval.callbacks import WandbCallback
from fseval.pipeline.estimator import Estimator
from omegaconf import MISSING
from sklearn.base import clone
from tqdm import tqdm

from .._experiment import Experiment
from ._config import RankAndValidatePipeline
from ._ranking_validator import RankingValidator
from ._subset_validator import SubsetValidator


@dataclass
class DatasetValidator(Experiment, RankAndValidatePipeline):
    """Validates an entire dataset, given a fitted ranker and its feature ranking. Fits
    at most `p` feature subsets, at each step incrementally including more top-features."""

    bootstrap_state: int = MISSING

    def _get_all_features_to_select(self, n: int, p: int) -> List[int]:
        """parse all features to select from config"""
        all_features_to_select_str = self.callbacks.config["all_features_to_select"]

        # set using `exec`
        localz = locals()
        exec(
            f"all_features_to_select = {all_features_to_select_str}",
            globals(),
            localz,
        )
        all_features_to_select = localz["all_features_to_select"]
        all_features_to_select = cast(List[int], all_features_to_select)
        all_features_to_select = list(all_features_to_select)

        assert (
            all_features_to_select
        ), f"Incorrect `all_features_to_select` string: {all_features_to_select_str}"

        return all_features_to_select

    def _get_estimator(self):
        all_features_to_select = self._get_all_features_to_select(
            self.dataset.n, self.dataset.p
        )

        # validate all subsets
        for n_features_to_select in all_features_to_select:
            config = self._get_config()
            validator = config.pop("validator")

            yield SubsetValidator(
                **config,
                validator=clone(validator),
                n_features_to_select=n_features_to_select,
                bootstrap_state=self.bootstrap_state,
            )

    def _get_estimator_repr(self, estimator):
        return Estimator._get_estimator_repr(estimator.validator)

    def _get_overrides_text(self, estimator):
        return f"[n_features_to_select={estimator.n_features_to_select}] "

    def score(self, X, y):
        scores = super(DatasetValidator, self).score(X, y)
        return scores


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

    def score(self, X, y):
        ranking_score = self.ranking_validator.score(X, y)
        ranking_score["group"] = "ranking"

        validation_score = self.dataset_validator.score(X, y)
        validation_score["group"] = "validation"

        scores = pd.DataFrame()
        scores = scores.append(ranking_score)
        scores = scores.append(validation_score)
        scores["bootstrap_state"] = self.bootstrap_state

        self.logger.info(f"scored bootstrap_state={self.bootstrap_state} ✓")
        return scores


@dataclass
class BootstrappedRankAndValidate(Experiment, RankAndValidatePipeline):
    """Provides an experiment that performs a 'bootstrap' procedure: using different
    `random_state` seeds the dataset is continuously resampled with replacement, such
    that various metrics can be better approximated."""

    logger: Logger = getLogger(__name__)

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

    def score(self, X, y):
        scores = super(BootstrappedRankAndValidate, self).score(X, y)
        self.storage_provider.save(
            "scores.csv", lambda file: scores.to_csv(file, index=False)
        )

        ranking_scores = scores[scores["group"] == "ranking"].dropna(axis=1)
        ranking_scores = ranking_scores.drop(columns=["group"])
        ranking_scores = ranking_scores.set_index("bootstrap_state")
        validation_scores = scores[scores["group"] == "validation"].dropna(axis=1)
        validation_scores = validation_scores.drop(columns=["group"])

        ##### Ranking scores - aggregation
        agg_ranking_scores = ranking_scores.agg(["mean", "std", "var", "min", "max"])
        # print scores
        self.logger.info(f"{self.ranker.name} ranking scores:")
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
        self.logger.info(f"{self.validator.name} validation scores:")
        agg_val_scores = val_scores_per_feature.mean().drop(columns=["bootstrap_state"])
        print(agg_val_scores)

        ##### Summary
        best = {}
        # best validator
        all_agg_val_scores = agg_val_scores.reset_index()
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
            wandb_callback = cast(WandbCallback, wandb_callback)

            ### upload summary as table: meta data and best scores
            # metadata
            config = self.callbacks.config
            metadata = dict()
            metadata["ranker.name"] = config["ranker"]["name"]
            metadata["validator.name"] = config["validator"]["name"]
            metadata["dataset.name"] = config["dataset"]["name"]
            metadata_df = pd.DataFrame([metadata])
            # best scores
            best_subset_prefixed = best_subset.add_prefix("validator.")
            best_ranker_prefixed = best_ranker.add_prefix("ranker.")
            best_scores = pd.concat([best_subset_prefixed, best_ranker_prefixed])
            best_scores_df = pd.DataFrame([best_scores])
            # upload tabular summary
            tabular_summary = best_scores_df.assign(**metadata_df)
            wandb_callback.upload_table(tabular_summary, "tabular_summary")

            ### upload ranking scores
            wandb_callback.upload_table(ranking_scores.reset_index(), "ranking_scores")

            ### upload validation scores
            wandb_callback.upload_table(validation_scores, "validation_scores")

        return summary
