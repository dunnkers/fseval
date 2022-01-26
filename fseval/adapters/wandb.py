from dataclasses import dataclass
from typing import List, Tuple

import wandb

from fseval.config.adapters import WandbDataset
from fseval.types import AbstractAdapter


@dataclass
class Wandb(AbstractAdapter, WandbDataset):
    def get_data(self) -> Tuple[List, List]:
        api = wandb.Api()
        artifact = api.artifact(self.artifact_id)
        X = artifact.get("X").data
        Y = artifact.get("Y").data
        return X, Y
