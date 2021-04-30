import hydra
import mleval   # must import at top level
import sklearn  # must import at top level
from omegaconf import DictConfig
from mleval.datasrc import DataSource
from mleval.tasks import Task
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from typing import Union

@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    datasrc: DataSource
    datasrc = hydra.utils.instantiate(cfg.datasrc)

    cv: BaseCrossValidator
    cv = hydra.utils.instantiate(cfg.cv)

    task: Task
    task = hydra.utils.instantiate(cfg.task, cfg=cfg, datasrc=datasrc, cv=cv)
    task.run()

if __name__ == "__main__":
    run()