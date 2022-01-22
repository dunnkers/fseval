from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class OpenMLDataset:
    _target_: str = "fseval.adapters.openml.OpenML"
    dataset_id: int = MISSING
    target_column: str = MISSING
    drop_qualitative: bool = False
