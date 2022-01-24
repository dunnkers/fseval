import os
import time
from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from fseval.types import Callback
from fseval.utils.uuid_utils import generate_shortuuid


@dataclass
class BaseExportCallback(Callback):
    """Provides base functionality for callbacks to export results."""

    def get_experiment_config(self, config: DictConfig) -> pd.DataFrame:
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

        # create dataframe with config
        df = pd.DataFrame([prepared_cfg])
        df = df.set_index("id")
        df["date_created"] = pd.Timestamp(time.time(), unit="s")

        return df

    def add_experiment_id(self, df: pd.DataFrame):
        assert hasattr(self, "id") and type(self.id) == str, (
            "Experiment does not have a shortuuid. Start of pipeline was "
            + "possibly unsuccessfully executed. Make sure `on_begin` is always called."
        )

        df["id"] = self.id
        df.set_index(["id"], append=True)

        return df
