import numpy as np
import pandas as pd
import pytest
from fseval.callbacks.to_sql import SQLCallback
from fseval.types import Task
from omegaconf import DictConfig, OmegaConf


def test_init_no_params():
    """Initialization should fail when no `engine` param was supplied."""
    # no `engine`
    with pytest.raises(AssertionError):
        SQLCallback()

    # no `engine.url`
    with pytest.raises(AssertionError):
        SQLCallback(engine={})


@pytest.fixture
def config() -> DictConfig:
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


@pytest.fixture
def to_sql() -> SQLCallback:
    to_sql = SQLCallback(engine={"url": "sqlite://"})
    return to_sql


def test_on_begin(to_sql: SQLCallback, config: DictConfig):
    """Test whether `on_begin` pipeline callback correctly stores experiment information
    in the `experiments` table."""

    # pretend `starting` pipeline
    to_sql.on_begin(config)

    # retrieve table using Pandas
    df: pd.DataFrame = pd.read_sql("experiments", con=to_sql.engine, index_col="id")

    # assert columns
    assert "dataset" in df.columns
    assert "dataset/n" in df.columns
    assert "dataset/p" in df.columns

    # assert types
    assert np.isscalar(df["dataset/n"][0])
    assert np.isscalar(df["dataset/p"][0])
    assert "datetime" in str(df.dtypes["date_created"])


def test_on_table(to_sql: SQLCallback, config: DictConfig):
    """Tests whether the callback successfully stores data in a table when calling the
    `on_table` callback."""

    # store dataframe in table
    df_to_store = pd.DataFrame([{"some_metric": 343}])
    name = "some_table"
    to_sql.on_begin(config)  # necessary to set `self.id`
    to_sql.on_table(df_to_store, name)

    # retrieve using Pandas
    df: pd.DataFrame = pd.read_sql("some_table", con=to_sql.engine, index_col="id")

    # assert table columns and record types
    assert "some_metric" in df.columns
    assert np.isscalar(df["some_metric"][0])
    assert len(df) == 1
