import numpy as np
import pytest
from fseval.datasets import Dataset
from hydra.utils import instantiate
from tests.integration.hydra_utils import TestGroupItem, generate_group_tests


def pytest_generate_tests(metafunc):
    generate_group_tests("dataset", metafunc)


class TestDataset(TestGroupItem):
    def test_initialization(self, cfg):
        ds = instantiate(cfg)
        assert isinstance(ds, Dataset)

    @pytest.fixture
    def ds(self, cfg):
        ds = instantiate(cfg)
        return ds

    def test_load(self, ds):
        ds.load()
        assert len(ds.X) > 0
        assert len(ds.y) > 0

    def test_feature_relevancy(self, ds):
        ds.load()
        if ds.feature_relevancy:
            assert ds.relevant_features is not None
            assert ds.relevant_features.shape == ds.X.shape
