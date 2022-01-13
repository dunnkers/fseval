from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import Callback


@dataclass
class SomeCallback(Callback):
    ran_on_begin: bool = False
    ran_on_config_update: bool = False
    ran_on_metrics: bool = False
    ran_on_table: bool = False
    ran_on_summary: bool = False
    ran_on_end: bool = False

    def on_begin(self, config: DictConfig):
        self.ran_on_begin = True

    def on_config_update(self, config: Dict):
        self.ran_on_config_update = True

    def on_metrics(self, metrics):
        self.ran_on_metrics = True

    def on_table(self, df: pd.DataFrame, name: str):
        self.ran_on_table = True

    def on_summary(self, summary: Dict):
        self.ran_on_summary = True

    def on_end(self, exit_code: Optional[int] = None):
        self.ran_on_end = True


def test_callback_iteration():
    some_callback: SomeCallback = SomeCallback()
    callback_collection: CallbackCollection = CallbackCollection(
        some_callback=some_callback
    )
    assert callback_collection.callback_names == ["some_callback"]
    assert len(list(callback_collection._iterator)) == 1

    # run callbacks
    callback_collection.on_begin(OmegaConf.create({}))
    callback_collection.on_config_update({})
    callback_collection.on_metrics({})
    callback_collection.on_table(pd.DataFrame(), "")
    callback_collection.on_summary({})
    callback_collection.on_end()

    # assert
    assert some_callback.ran_on_begin == True
    assert some_callback.ran_on_config_update == True
    assert some_callback.ran_on_metrics == True
    assert some_callback.ran_on_table == True
    assert some_callback.ran_on_summary == True
    assert some_callback.ran_on_end == True
