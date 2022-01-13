import tempfile
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import pytest
from fseval.callbacks.to_excel import ExcelCallback
from fseval.types import Callback, Task
from omegaconf import DictConfig, OmegaConf
from overrides import overrides

from ._common import BaseCallbackTest


class TestExcelCallback(BaseCallbackTest):
    @overrides
    def get_callback(self) -> Callback:
        # Create temporary dir. Callback should support nested directories - and create
        # them when they do not exist yet accordingly.
        filepath = Path(tempfile.mkdtemp())
        filepath = filepath / "some_sub_dir"
        filepath = filepath / "some_file.xlsx"

        # setup callback
        callback = ExcelCallback(filepath=filepath)

        return callback

    @overrides
    def restore_config(self, callback: Callback) -> pd.DataFrame:
        callback = cast(ExcelCallback, callback)
        filepath: Path = callback.filepath
        df: pd.DataFrame = pd.read_excel(
            filepath,
            sheet_name="experiments",
            index_col=0,
            dtype={"date_created": "datetime"},
        )

        return df

    @overrides
    def restore_table(self, callback: Callback, table_name: str) -> pd.DataFrame:
        callback = cast(ExcelCallback, callback)
        filepath: Path = callback.filepath
        df: pd.DataFrame = pd.read_excel(filepath, index_col=0, sheet_name=table_name)

        return df

    def test_init(self):
        """Initialization should fail when no `filepath` param was supplied."""
        # no `filepath`
        with pytest.raises(AssertionError):
            ExcelCallback()

        # assert instantiation in __init__ was successfull
        callback = self.get_callback()
        assert isinstance(callback, ExcelCallback)
