from typing import Dict, List, cast

import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.base import AbstractEstimator
from fseval.callbacks._callback import CallbackList
from fseval.config import BaseConfig
from fseval.cv import CrossValidator
from fseval.datasets import Dataset


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    dataset: Dataset = instantiate(cfg.dataset)  # pipeline.
    cv: CrossValidator = instantiate(cfg.cv)  # pipeline.
    callbacks = instantiate(cfg.callbacks)
    callback_list = CallbackList(callbacks.values())
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    primitive_cfg = cast(Dict, primitive_cfg)
    callback_list.set_config(primitive_cfg)

    # begin
    callback_list.on_begin()

    # load dataset and split
    dataset.load(ensure_positive_X=True)
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    # run pipeline
    pipeline: AbstractEstimator = instantiate(
        cfg.pipeline, callback_list=callback_list, p=np.shape(X_train)[1]
    )
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    callback_list.on_metrics(scores)

    # end
    callback_list.on_end()


if __name__ == "__main__":
    main()
