from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict

import pandas as pd
from omegaconf import MISSING, DictConfig
from sqlalchemy import create_engine

from fseval.config.callbacks.to_sql import ToSQLCallback
from fseval.types import TerminalColor

from ._base_export_callback import BaseExportCallback


@dataclass
class SQLCallback(BaseExportCallback, ToSQLCallback):
    """SQL support for fseval. Uploads general information on the experiment to
    a `experiments` table and provides a hook for uploading custom tables. Use the
    `on_table` hook in your pipeline to upload a DataFrame to a certain database table.

    Support for SQL exports is achieved through using Pandas `df.to_sql` function. This
    function, in its turn, then uses SQLAlchemy to export to SQL. Therefore, to use this
    callback, it is required you configure the `engine.url` parameter, used to connect
    with the database.
    """

    def __post_init__(self):
        # assert SQL Alchemy config
        assert self.url != MISSING, (
            "The SQL callback did not receive a `url` param. "
            + "This is required to set up SQLAlchemy."
        )
        assert isinstance(self.kwargs, Dict)

        # log - tell user callback is enabled
        self.logger: Logger = getLogger(__name__)
        self.logger.info("SQL callback enabled.")

    def on_begin(self, config: DictConfig):
        # create SQL engine
        self.engine = create_engine(self.url, **self.kwargs)

        # upload experiment config to SQL database
        df = self.get_experiment_config(config)
        df.to_sql("experiments", con=self.engine, if_exists=self.if_table_exists)
        self.logger.info(
            f"Written experiment config to {TerminalColor.blue('experiments')} SQL table "
            + f"{TerminalColor.green('✓')}"
        )

    def on_table(self, df: pd.DataFrame, name: str):
        # make sure experiment `id` is added to this table. this allows a user to JOIN
        # the results back into each other, after being distributed over several
        # database tables.
        df = self.add_experiment_id(df)

        # upload table to SQL database
        df.to_sql(name, con=self.engine, if_exists=self.if_table_exists)
        self.logger.info(
            f"Uploaded results to {TerminalColor.blue(name)} SQL table "
            + f"{TerminalColor.green('✓')}"
        )
