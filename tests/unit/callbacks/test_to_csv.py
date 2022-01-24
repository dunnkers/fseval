import tempfile
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from overrides import overrides

from fseval.callbacks.to_csv import CSVCallback
from fseval.types import Callback

from ._common import BaseCallbackTest


class TestCSVCallback(BaseCallbackTest):
    @overrides
    def get_callback(self) -> Callback:
        # Create temporary dir. Callback should support nested directories - and create
        # them when they do not exist yet accordingly.
        csv_dir = Path(tempfile.mkdtemp())
        csv_dir = csv_dir / "some_sub_dir"

        # setup callback
        callback = CSVCallback(dir=str(csv_dir))

        return callback

    @overrides
    def restore_config(self, callback: Callback) -> pd.DataFrame:
        callback = cast(CSVCallback, callback)
        filepath = callback.save_dir / "experiments.csv"
        df: pd.DataFrame = pd.read_csv(
            filepath,
            index_col="id",
            parse_dates=["date_created"],
        )

        return df

    @overrides
    def restore_table(self, callback: Callback, table_name: str) -> pd.DataFrame:
        callback = cast(CSVCallback, callback)
        filepath = callback.save_dir / f"{table_name}.csv"
        df: pd.DataFrame = pd.read_csv(filepath, index_col="id")

        return df

    def test_init(self):
        """Initialization should fail when no `dir` param was supplied."""
        # no `dir`
        with pytest.raises(AssertionError):
            CSVCallback()

        # assert instantiation in __init__ was successfull
        callback = self.get_callback()
        assert isinstance(callback, CSVCallback)
