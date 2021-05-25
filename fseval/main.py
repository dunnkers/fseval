import copy
import sys
import traceback
from logging import Logger, getLogger
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
from fseval.types import AbstractPipeline, AbstractStorageProvider, IncompatibilityError


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    logger = getLogger(__name__)

    # instantiate and load dataset
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()

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

    # instantiate cv
    cv: CrossValidator = instantiate(cfg.cv)

    # instantiate pipeline
    try:
        pipeline: AbstractPipeline = instantiate(
            cfg.pipeline, callbacks, dataset, cv, storage_provider
        )
    except IncompatibilityError as e:
        traceback.print_exc()
        (msg,) = e.args
        logger.warn(
            f"encountered expected pipeline incompatibility with current config:"
        )
        logger.error(msg)
        logger.warn("exiting gracefully...")
        sys.exit(0)

    # run pipeline
    callbacks.on_begin()
    dataset.print_dataset_details()
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    callbacks.on_summary(scores)
    callbacks.on_end()


if __name__ == "__main__":
    main()
