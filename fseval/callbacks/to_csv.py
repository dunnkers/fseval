import os
import time
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ._base_export_callback import BaseExportCallback


class CSVCallback(BaseExportCallback):
    """CSV support for fseval. Uploads general information on the experiment to
    a `experiments` table and provides a hook for uploading custom tables. Use the
    `on_table` hook in your pipeline to upload a DataFrame to a certain database table.
    """

    def __init__(self, **kwargs):
        super(CSVCallback, self).__init__()

        # extract config
        self.dir = kwargs.get("dir")
        # self.if_table_exists = kwargs.get("if_table_exists", "append")

        # assert dir param was given
        assert self.dir, (
            "The CSV callback did not receive a `dir` param. All results will be "
            + "written to files in this dir. This is required to export to CSV files."
        )

        # upgrade dir to Path type
        self.dir = Path(self.dir)

    def on_begin(self, config: DictConfig):
        df = self.get_experiment_config(config)

        # write experiment config to `experiments.csv`
        df.to_csv(self.dir / "experiments.csv")

    def on_table(self, df: pd.DataFrame, name: str):
        # make sure experiment `id` is added to this table. this allows a user to JOIN
        # the results back into each other, after being distributed over several
        # database tables.
        df = self.add_experiment_id(df)

        # upload table to CSV database
        df.to_csv(self.dir / f"{name}.csv")
