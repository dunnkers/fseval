import os
import time

import pandas as pd
from fseval.types import Callback
from fseval.utils.uuid_utils import generate_shortuuid
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import create_engine


class SQLAlchemyCallback(Callback):
    """SQLAlchemy support for fseval. Uploads general information on the experiment to
    a `experiments` table and provides a hook for uploading custom tables. Use the
    `on_table` hook in your pipeline to upload a DataFrame to a certain database table.
    """

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
            "dataset/n": config.dataset.n,
            "dataset/p": config.dataset.p,
            "dataset/task": config.dataset.task.name,  # `.name` because Enum
            "dataset/group": config.dataset.group,
            "dataset/domain": config.dataset.domain,
            "ranker": config.ranker.name,
            "validator": config.validator.name,
            "local_dir": os.getcwd(),
        }

        # add random id
        self.id: str = generate_shortuuid()
        prepared_cfg["id"] = self.id

        # create SQL engine
        self.engine = create_engine(**self.engine_kwargs)

        # upload experiment config to database
        df = pd.DataFrame([prepared_cfg])
        df = df.set_index("id")
        df["date_created"] = pd.Timestamp(time.time(), unit="s")

        df.to_sql("experiments", con=self.engine, if_exists=self.if_table_exists)

    def on_table(self, df: pd.DataFrame, name: str):
        assert hasattr(self, "id") and type(self.id) == str, (
            "No database shortuuid. SQL Alchemy callback was not properly invoked at "
            + "the start of the pipeline. Make sure `on_begin` is always called."
        )

        df["id"] = self.id
        df.set_index(["id"], append=True)
        df.to_sql(name, con=self.engine, if_exists=self.if_table_exists)
