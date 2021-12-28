import numpy as np
import pytest
from fseval.config import PipelineConfig
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.utils.hydra_utils import get_group_pipeline_configs
from hydra.utils import instantiate

from ._group_test_utils import ShouldTestGroupItem


def pytest_generate_tests(metafunc):
    argvalues, pytest_ids = get_group_pipeline_configs(
        config_module="tests.integration.conf",
        config_name="simple_defaults",
        group_name="dataset",
        should_test=metafunc.cls.should_test,
    )
    metafunc.parametrize(
        "cfg",
        argvalues,
        ids=pytest_ids,
        scope="class",
    )


class DatasetTest(ShouldTestGroupItem):
    __test__ = False

    @pytest.fixture
    def ds_loader(self, cfg: PipelineConfig) -> DatasetLoader:
        ds_loader: DatasetLoader = instantiate(cfg.dataset)
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
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        return cfg.dataset.feature_importances is not None

    def test_feature_importances(self, ds_loader):
        ds: Dataset = ds_loader.load()
        X_importances = ds.feature_importances

        assert np.ndim(X_importances) == 1 or np.ndim(X_importances) == 2

        # ensure importance is defined as a probability vector
        if np.ndim(X_importances) == 1:
            assert np.isclose(X_importances.sum(), 1.0)
        if np.ndim(X_importances) == 2:
            assert np.isclose(X_importances.sum(axis=1), 1.0).all()
