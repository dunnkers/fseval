from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class StorageConfig:
    _target_: str = MISSING
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None
