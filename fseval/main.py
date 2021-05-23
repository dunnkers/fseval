import copy
from typing import Dict, cast

import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.config import BaseConfig
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import AbstractEstimator, AbstractStorageProvider


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    # convert to primitive dict
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    primitive_cfg = cast(Dict, primitive_cfg)

    # instantiate callback collection
    callbacks = instantiate(cfg.callbacks)
    callbacks = CallbackCollection(callbacks)

    # prepare and set config object on callbacks: put everything in `pipeline` root
    prepared_cfg = copy.deepcopy(primitive_cfg)
    pipeline_cfg = prepared_cfg.pop("pipeline")
    prepared_cfg = {**pipeline_cfg, **prepared_cfg}
    prepared_cfg["pipeline"] = prepared_cfg.pop("name")
    callbacks.set_config(prepared_cfg)

    # instantiate storage provider
    storage_provider: AbstractStorageProvider = instantiate(cfg.storage_provider)
    storage_provider.set_config(primitive_cfg)

    # instantiate dataset and cv
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    cv: CrossValidator = instantiate(cfg.cv)

    # begin: load dataset and split
    callbacks.on_begin()
    dataset: Dataset = dataset_loader.load()
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    # instantiate pipeline
    pipeline: AbstractEstimator = instantiate(
        cfg.pipeline, callbacks, dataset, cv, storage_provider
    )

    # run pipeline, send metrics and finish
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    callbacks.on_summary(scores)
    callbacks.on_end()


if __name__ == "__main__":
    main()
