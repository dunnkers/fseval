from copy import deepcopy
from logging import getLogger
from traceback import print_exc
from typing import Dict, cast

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.config import BaseConfig
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import (
    AbstractPipeline,
    AbstractStorageProvider,
    IncompatibilityError,
    TerminalColor,
)


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    logger = getLogger(__name__)
    logger.info("instantiating pipeline components...")

    # instantiate and load dataset. set cfg runtime properties afterwards.
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()
    cfg.dataset.n = dataset.n
    cfg.dataset.p = dataset.p
    cfg.dataset.multioutput = dataset.multioutput

    # convert to primitive dict
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    primitive_cfg = cast(Dict, primitive_cfg)

    # instantiate callback collection
    callbacks = instantiate(cfg.callbacks)
    callbacks = CallbackCollection(callbacks)

    # prepare and set config object on callbacks: put everything in `pipeline` root
    prepared_cfg = deepcopy(primitive_cfg)
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
    logger.info(f"instantiating pipeline...")
    try:
        pipeline: AbstractPipeline = instantiate(
            cfg.pipeline, callbacks, dataset, cv, storage_provider
        )
    except IncompatibilityError as e:
        (msg,) = e.args
        logger.error(msg)
        logger.info(
            "encountered an expected pipeline incompatibility with the current config, "
            + "exiting gracefully..."
        )
        return

    # run pipeline
    pipeline_name = getattr(pipeline, "name", "")
    logger.info(f"starting {pipeline_name} pipeline...")
    callbacks.on_begin()
    logger.info(
        f"using dataset: {TerminalColor.yellow(dataset_loader.name)} "
        + f"[{dataset._log_details}]"
    )
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print_exc()
        logger.error(e)
        logger.info(
            "error occured during pipeline fitting step... "
            + "exiting with a status code 1."
        )
        callbacks.on_end(exit_code=1)
        raise e

    scores = pipeline.score(X_test, y_test)
    logger.info(f"{pipeline_name} pipeline finished {TerminalColor.green('✓')}")
    callbacks.on_summary(scores)
    callbacks.on_end()


if __name__ == "__main__":
    main()
