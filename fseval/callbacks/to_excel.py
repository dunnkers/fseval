import os
import time
from logging import Logger, getLogger
from pathlib import Path

import pandas as pd
from fseval.types import TerminalColor
from omegaconf import DictConfig, OmegaConf

from .to_csv import CSVCallback


class ExcelCallback(CSVCallback):
    """Excel support for fseval. Writes tables to a single excel file as worksheets.
    The experiment config is written to a `experiments` worksheet.
    """

    def __init__(self, **kwargs):
        # extract config
        self.filepath = kwargs.get("filepath")
        self.mode = kwargs.get("mode", "a")

        # assert filepath param was given
        assert self.filepath, (
            "The Excel callback did not receive a `filepath` param. All results will be "
            + "written to files in this filepath. This is required to export to an Excel file."
        )

        # create dirs where necessary
        self.filepath = Path(self.filepath)
        parent_dir = self.filepath.parent

        if not parent_dir.is_dir():  # ensure directories exist
            parent_dir.mkdir(parents=True)  # parents=True so creates recursively

        # print save path
        filepath_abs_str = TerminalColor.blue(self.filepath.absolute())
        self.logger: Logger = getLogger(__name__)
        self.logger.info(
            f"Excel callback enabled. Writing worksheets to: {filepath_abs_str}"
        )

    def on_begin(self, config: DictConfig):
        df = self.get_experiment_config(config)

        # write experiment config to `experiments` worksheet
        header = self.should_insert_header(self.filepath)
        mode = self.mode if self.filepath.exists() else "w"
        if_sheet_exists = "new" if mode == "a" else None
        with pd.ExcelWriter(
            self.filepath, mode=mode, if_sheet_exists=if_sheet_exists
        ) as writer:
            df.to_excel(writer, sheet_name="experiments", header=header)

        # log
        filepath_abs_str = TerminalColor.blue(self.filepath.absolute())
        self.logger.info(
            f"Written experiment config to: {filepath_abs_str} {TerminalColor.green('✓')}"
        )

    def on_table(self, df: pd.DataFrame, name: str):
        # make sure experiment `id` is added to this table. this allows a user to JOIN
        # the results back into each other, after being distributed over several
        # database tables.
        df = self.add_experiment_id(df)

        # upload table to Excel worksheet named after the table name
        header = self.should_insert_header(self.filepath)
        mode = self.mode if self.filepath.exists() else "w"
        with pd.ExcelWriter(self.filepath, mode=mode, if_sheet_exists="new") as writer:
            df.to_excel(writer, sheet_name=name, header=header)

        # log table upload
        filepath_abs_str = TerminalColor.blue(self.filepath.absolute())
        self.logger.info(
            f"Written `{name}` worksheet to: {filepath_abs_str} {TerminalColor.green('✓')}"
        )
