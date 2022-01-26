from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path

import pandas as pd
from omegaconf import MISSING, DictConfig

from fseval.config.callbacks.to_csv import ToCSVCallback
from fseval.types import TerminalColor

from ._base_export_callback import BaseExportCallback


@dataclass
class CSVCallback(BaseExportCallback, ToCSVCallback):
    """CSV support for fseval. Uploads general information on the experiment to
    a `experiments` table and provides a hook for uploading custom tables. Use the
    `on_table` hook in your pipeline to upload a DataFrame to a certain database table.
    """

    def __post_init__(self):
        # assert dir param was given
        assert self.dir != MISSING, (
            "The CSV callback did not receive a `dir` param. All results will be "
            + "written to files in this dir. This is required to export to CSV files."
        )

        # upgrade dir to Path type
        self.save_dir = Path(self.dir)

        # create directories where necessary
        if not self.save_dir.is_dir():  # ensure directories exist
            self.save_dir.mkdir(parents=True)  # parents=True so creates recursively

        # print save path
        dir_abs_str = TerminalColor.blue(self.save_dir.absolute())
        self.logger: Logger = getLogger(__name__)
        self.logger.info(f"CSV callback enabled. Writing .csv files to: {dir_abs_str}")

    def should_insert_header(self, filepath: Path) -> bool:
        if filepath.exists():
            # when the target `.csv` file already exists, omit header.
            return False
        else:
            # otherwise, add a header to the csv file.
            return True

    def on_begin(self, config: DictConfig):
        df = self.get_experiment_config(config)

        # write experiment config to `experiments.csv`
        filepath = self.save_dir / "experiments.csv"
        header = self.should_insert_header(filepath)
        df.to_csv(filepath, mode=self.mode, header=header)

        # log
        filepath_abs_str = TerminalColor.blue(filepath.absolute())
        self.logger.info(
            f"Written experiment config to: {filepath_abs_str} {TerminalColor.green('✓')}"
        )

    def on_table(self, df: pd.DataFrame, name: str):
        # make sure experiment `id` is added to this table. this allows a user to JOIN
        # the results back into each other, after being distributed over several
        # database tables.
        df = self.add_experiment_id(df)

        # upload table to CSV file, named after the table name
        filepath = self.save_dir / f"{name}.csv"
        header = self.should_insert_header(filepath)
        df.to_csv(filepath, mode=self.mode, header=header)

        # log table upload
        filepath_abs_str = TerminalColor.blue(filepath.absolute())
        self.logger.info(
            f"Written `{name}` table to: {filepath_abs_str} {TerminalColor.green('✓')}"
        )
