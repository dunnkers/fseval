from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from fseval.types import Callback, Task
from omegaconf import DictConfig, OmegaConf


class BaseCallbackTest(ABC):
    @abstractmethod
    def get_callback(self) -> Callback:
        ...

    @abstractmethod
    def restore_config(self, callback: Callback) -> pd.DataFrame:
        ...

    @abstractmethod
    def restore_table(self, callback: Callback, table_name: str) -> pd.DataFrame:
        ...

    @pytest.fixture(scope="module")
    def config(self) -> DictConfig:
        config: DictConfig = OmegaConf.create(
            {
                "dataset": {
                    "name": "some_dataset",
                    "n": 10000,
                    "p": 5,
                    "task": Task.classification,
                    "group": "some_group",
                    "domain": "some_domain",
                },
                "ranker": {
                    "name": "some_ranker",
                },
                "validator": {"name": "some_validator"},
            }
        )

        return config

    def test_on_begin(
        self,
        config: DictConfig,
    ):
        """Test whether `on_begin` pipeline callback correctly stores experiment
        information. Then asserts whether the experiment config as defined in
        `conftest.py` was successfully reconstructed."""

        # pretend `starting` pipeline
        callback = self.get_callback()
        callback.on_begin(config)

        # retrieve table
        df: pd.DataFrame = self.restore_config(callback)

        ## assert experiment config
        # assert columns
        assert "id" == df.index.name
        assert "dataset" in df.columns
        assert "dataset/n" in df.columns
        assert "dataset/p" in df.columns

        # assert types
        assert np.isscalar(df["dataset/n"][0])
        assert np.isscalar(df["dataset/p"][0])
        assert "datetime" in str(df.dtypes["date_created"])

    def test_on_table(
        self,
        config: DictConfig,
    ):
        """Tests whether the callback successfully operates when calling the
        `on_table` callback. Then asserts whether the `df_to_store` DataFrame defined
        in `conftest.py` was successfully reconstructed by the callback."""

        # pretend `starting` pipeline
        callback = self.get_callback()
        callback.on_begin(config)  # necessary to set `self.id`

        # 1) store dataframe in table
        table_name = "some_table"
        df_to_store_1 = pd.DataFrame([{"some_metric": 343}])
        callback.on_table(df_to_store_1, table_name)

        # 2) store another dataframe. this table should not **replace** the existing
        # data - rather, it should be appended.
        df_to_store_2 = pd.DataFrame([{"some_metric": 400}])
        callback.on_table(df_to_store_2, table_name)

        # restore
        df = self.restore_table(callback, table_name)

        # assert table columns and record types
        assert "id" == df.index.name
        assert "some_metric" in df.columns
        assert np.isscalar(df["some_metric"][0])
        assert len(df) == 2
