import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fseval.callbacks.to_csv import CSVCallback
from fseval.types import Task
from omegaconf import DictConfig, OmegaConf


def test_init_no_params():
    """Initialization should fail when no `dir` param was supplied."""
    # no `dir`
    with pytest.raises(AssertionError):
        CSVCallback()

    # should work
    instance = CSVCallback(dir="some_dir")
    assert isinstance(instance, CSVCallback)


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


def test_on_begin(config: DictConfig):
    """Test whether `on_begin` pipeline callback correctly stores experiment information
    in a csv file called `experiments.csv`."""

    tmpdir = Path(tempfile.mkdtemp())
    to_csv = CSVCallback(dir=tmpdir)

    # pretend `starting` pipeline
    to_csv.on_begin(config)

    # retrieve table using Pandas
    df: pd.DataFrame = pd.read_csv(
        tmpdir / "experiments.csv", index_col="id", parse_dates=["date_created"]
    )

    # assert columns
    assert "id" == df.index.name
    assert "dataset" in df.columns
    assert "dataset/n" in df.columns
    assert "dataset/p" in df.columns

    # assert types
    assert np.isscalar(df["dataset/n"][0])
    assert np.isscalar(df["dataset/p"][0])
    assert "datetime" in str(df.dtypes["date_created"])


def test_on_table(config: DictConfig):
    """Tests whether the callback successfully stores data in csv files when calling the
    `on_table` callback."""

    # create temporary dir, with one nesting level. this ensures the callback is able
    # to create subdirectories by itself.
    tmpdir = Path(tempfile.mkdtemp()) / "some_sub_dir"
    to_csv = CSVCallback(dir=tmpdir)

    # store dataframe in table
    df_to_store = pd.DataFrame([{"some_metric": 343}])
    name = "some_table"
    to_csv.on_begin(config)  # necessary to set `self.id`
    to_csv.on_table(df_to_store, name)

    # this table should be appended to the csv file, not replacing the file.
    df_to_store = pd.DataFrame([{"some_metric": 400}])
    to_csv.on_table(df_to_store, name)

    # retrieve using Pandas
    filepath: Path = tmpdir / "some_table.csv"
    df: pd.DataFrame = pd.read_csv(filepath, index_col="id")

    # assert table columns and record types
    assert "id" == df.index.name
    assert "some_metric" in df.columns
    assert np.isscalar(df["some_metric"][0])
    assert len(df) == 2
