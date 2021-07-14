import copy
import sys
import time
from collections import UserDict
from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Optional, cast

import pandas as pd
import pangres
from fseval.types import Callback
from fseval.utils.dict_utils import dict_flatten, dict_merge
from omegaconf import DictConfig, OmegaConf
from shortuuid import ShortUUID
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text


def generate_id():
    """Generate a random experiment ID."""
    characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
    run_gen = ShortUUID(alphabet=characters)
    return run_gen.random(8)


class SQLAlchemyCallback(Callback):
    def __init__(self, **kwargs):
        super(SQLAlchemyCallback, self).__init__()
        # make sure any nested objects are casted from DictConfig's to regular dict's.
        kwargs = OmegaConf.create(kwargs)
        kwargs = OmegaConf.to_container(kwargs)

        # assert SQL Alchemy config
        self.engine_kwargs = kwargs.get("engine")
        self.if_table_exists = kwargs.get("if_table_exists", "append")

        assert (
            self.engine_kwargs
        ), "The SQL Alchemy callback did not receive a `engine` param."

        assert self.engine_kwargs.get(
            "url"
        ), "The SQL Alchemy callback did not receive a `engine.url` param."

    def on_begin(self, config: DictConfig):
        prepared_cfg = {
            "dataset": config.dataset.name,
            "ranker": config.ranker.name,
            "validator": config.validator.name,
        }

        # add random id
        self.id = generate_id()
        prepared_cfg["id"] = self.id

        # create SQL engine
        self.engine = create_engine(**self.engine_kwargs)

        # upload experiment config to database
        df = pd.DataFrame([prepared_cfg])
        df = df.set_index("id")
        df["date_created"] = pd.Timestamp(time.time(), unit="s")

        # df.to_sql("experiments", con=self.engine, if_exists=self.if_table_exists)
        pangres.upsert(
            self.engine,
            df,
            "experiments",
            if_row_exists="update",
            add_new_columns=True,
            adapt_dtype_of_empty_db_columns=True,
        )

    def on_table(self, df: pd.DataFrame, name: str):
        df["id"] = self.id
        # df = df.set_index("id")
        df = df.reset_index(drop=True)
        # df.to_sql(name, con=self.engine, if_exists=self.if_table_exists)
        pangres.upsert(
            self.engine,
            df,
            name,
            if_row_exists="update",
            add_new_columns=True,
            adapt_dtype_of_empty_db_columns=True,
        )

    def on_config_update(self, config: Dict):
        ...

    def on_metrics(self, metrics):
        ...

    def on_summary(self, summary: Dict):
        ...

    def on_end(self, exit_code: Optional[int] = None):
        ...
