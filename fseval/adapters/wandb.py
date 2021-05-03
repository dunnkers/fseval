from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from omegaconf import MISSING

import wandb

from ._adapter import Adapter


@dataclass
class Wandb(Adapter):
    artifact_id: str = MISSING

    def get_data(self) -> Tuple[List, List]:
        api = wandb.Api()
        artifact = api.artifact(self.artifact_id)
        X = artifact.get("X").data
        Y = artifact.get("Y").data
        return X, Y
