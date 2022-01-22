from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class WandbDataset:
    _target_: str = "fseval.adapters.wandb.Wandb"
    artifact_id: str = MISSING
