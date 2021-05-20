from typing import Dict, cast

import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.config import BaseConfig
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.pipelines._callback_list import CallbackList
from fseval.types import AbstractEstimator


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    # convert to primitive dict
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    primitive_cfg = cast(Dict, primitive_cfg)

    # instantiate callback list
    callbacks = instantiate(cfg.callbacks)
    callback_list = CallbackList(callbacks.values())
    callback_list.set_config(primitive_cfg)

    # instantiate dataset and cv
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    cv: CrossValidator = instantiate(cfg.cv)

    # begin: load dataset and split
    callback_list.on_begin()
    dataset: Dataset = dataset_loader.load()
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    # instantiate pipeline
    pipeline: AbstractEstimator = instantiate(cfg.pipeline, callback_list, dataset, cv)

    # run pipeline, send metrics and finish
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    callback_list.on_metrics(scores)
    callback_list.on_end()


if __name__ == "__main__":
    main()
