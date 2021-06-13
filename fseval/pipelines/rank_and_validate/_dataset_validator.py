from dataclasses import dataclass
from typing import List, cast

from omegaconf import MISSING
from sklearn.base import clone

from fseval.pipeline.estimator import Estimator

from .._experiment import Experiment
from ._config import RankAndValidatePipeline
from ._subset_validator import SubsetValidator


@dataclass
class DatasetValidator(Experiment, RankAndValidatePipeline):
    """Validates an entire dataset, given a fitted ranker and its feature ranking. Fits
    at most `p` feature subsets, at each step incrementally including more top-features."""

    bootstrap_state: int = MISSING

    def _get_all_features_to_select(self, n: int, p: int) -> List[int]:
        """parse all features to select from config"""
        assert n is not None and p is not None, "dataset must be loaded!"
        assert (
            n > 0 and p > 0
        ), f"dataset must have > 0 samples (n was {n} and p was {p})"

        # set using `exec`
        localz = locals()
        exec(
            f"all_features_to_select = {self.all_features_to_select}",
            globals(),
            localz,
        )
        all_features_to_select = localz["all_features_to_select"]
        all_features_to_select = cast(List[int], all_features_to_select)
        all_features_to_select = list(all_features_to_select)

        assert (
            all_features_to_select
        ), f"Incorrect `all_features_to_select` string: {self.all_features_to_select}"

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

    def score(self, X, y, **kwargs):
        scores = super(DatasetValidator, self).score(X, y, **kwargs)
        return scores
