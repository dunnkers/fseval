import pandas as pd

from fseval.types import (AbstractAdapter, AbstractEstimator, AbstractPipeline,
                          AbstractStorage, Callback)


def test_abstract_estimator():
    AbstractEstimator.__abstractmethods__ = set()
    instance = AbstractEstimator()
    assert instance.fit([], []) is None
    assert instance.score([], []) is None


def test_abstract_adapter():
    AbstractAdapter.__abstractmethods__ = set()
    instance = AbstractAdapter()
    assert instance.get_data() is None


def test_abstract_callback():
    Callback.__abstractmethods__ = set()
    instance = Callback()
    assert instance.on_begin({}) is None
    assert instance.on_config_update({}) is None
    assert instance.on_metrics({}) is None
    assert instance.on_table(pd.DataFrame(), "") is None
    assert instance.on_summary({}) is None
    assert instance.on_end() is None


def test_abstract_storage():
    AbstractStorage.__abstractmethods__ = set()
    instance = AbstractStorage()
    assert instance.get_load_dir() is None
    assert instance.get_save_dir() is None
    assert instance.save("", lambda: None) is None
    assert instance.save_pickle("", {}) is None
    assert instance.restore("", lambda: None) is None
    assert instance.restore_pickle("") is None


def test_abstract_pipeline():
    AbstractPipeline.__abstractmethods__ = set()
    instance = AbstractPipeline()
    assert instance.prefit() is None
    assert instance.postfit() is None
