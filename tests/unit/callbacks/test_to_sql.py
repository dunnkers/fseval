from typing import cast

import pandas as pd
import pytest
from fseval.callbacks.to_sql import SQLCallback
from fseval.config.callbacks.to_sql import EngineConfig, ToSQLCallback
from fseval.types import Callback
from overrides import overrides

from ._common import BaseCallbackTest


class TestSQLCallback(BaseCallbackTest):
    @overrides
    def get_callback(self) -> Callback:
        self.callback = SQLCallback(
            engine_config={"url": "sqlite://"}  # type: ignore
        )  # in-memory database
        return self.callback

    @overrides
    def restore_config(self, callback: Callback) -> pd.DataFrame:
        callback = cast(SQLCallback, callback)
        df: pd.DataFrame = pd.read_sql(
            "experiments", con=callback.engine, index_col="id"
        )
        return df

    @overrides
    def restore_table(self, callback: Callback, table_name: str) -> pd.DataFrame:
        callback = cast(SQLCallback, callback)
        df: pd.DataFrame = pd.read_sql(table_name, con=callback.engine, index_col="id")
        return df

    def test_init(self):
        """Initialization should fail when no `engine` param was supplied."""
        # no `engine`
        with pytest.raises(AssertionError):
            SQLCallback()

        # no `engine.url`
        with pytest.raises(AttributeError):
            SQLCallback(engine_config={})  # type: ignore
