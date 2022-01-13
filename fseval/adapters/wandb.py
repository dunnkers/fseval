from dataclasses import dataclass
from typing import List, Tuple

from omegaconf import MISSING

import wandb
from fseval.types import AbstractAdapter


@dataclass
class WandbDataset:
    _target_: str = "fseval.adapters.wandb.Wandb"
    artifact_id: str = MISSING


@dataclass
class Wandb(AbstractAdapter, WandbDataset):
    def get_data(self) -> Tuple[List, List]:
        api = wandb.Api()
        artifact = api.artifact(self.artifact_id)
        X = artifact.get("X").data
        Y = artifact.get("Y").data
        return X, Y
