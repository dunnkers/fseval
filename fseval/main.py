from typing import List, cast

import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

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

    dataset.load()
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    pipeline: Pipeline = instantiate(
        cfg.pipeline, callback_list=callback_list, p=np.shape(X_train)[1]
    )
    pipeline.fit(X_train, y_train)
    scores = pipeline.score(X_test, y_test)
    print(scores)


if __name__ == "__main__":
    main()
