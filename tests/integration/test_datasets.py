from typing import Optional

import numpy as np
import pytest
from fseval.pipeline.dataset import Dataset, DatasetLoader
from hydra.utils import instantiate
from omegaconf import DictConfig
from fseval.utils.hydra_utils import TestGroupItem, generate_group_tests


def pytest_generate_tests(metafunc):
    generate_group_tests("dataset", metafunc)


class DatasetTest(TestGroupItem):
    __test__ = False

    @pytest.fixture
    def ds_loader(self, cfg):
        ds_loader = instantiate(cfg)
        assert isinstance(ds_loader, DatasetLoader)
        return ds_loader


class TestAllDatasets(DatasetTest):
    __test__ = True

    def test_load(self, ds_loader):
        ds: Dataset = ds_loader.load()
        assert len(ds.X) > 0
        assert len(ds.y) > 0


class TestFeatureImportancesDatasets(DatasetTest):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        if cfg.feature_importances:
            return cfg
        else:
            return None

    def test_feature_importances(self, ds_loader):
        ds: Dataset = ds_loader.load()
        X_importances = ds.feature_importances

        assert np.ndim(X_importances) == 1 or np.ndim(X_importances) == 2

        # ensure importance is defined as a probability vector
        if np.ndim(X_importances) == 1:
            assert np.isclose(X_importances.sum(), 1.0)
        if np.ndim(X_importances) == 2:
            assert np.isclose(X_importances.sum(axis=1), 1.0).all()
