from hydra.utils import instantiate
from fseval.ranker.chi2 import Chi2
from typing import List
from fseval.config import RankerConfig
import pytest
from omegaconf import OmegaConf

@pytest.fixture(scope='module', autouse=True)
def ranker():
    ranker_cfg = RankerConfig(
        _target_='fseval.ranker.chi2.Chi2',
        name='Chi Squared',
        n_features_to_select=10,
        compatibility=['multiclass', 'multivariate']
    )
    cfg = OmegaConf.create(ranker_cfg)
    ranker = instantiate(cfg)
    return ranker


def test_initialization(ranker):
    assert ranker.name == 'Chi Squared'
    assert len(ranker.compatibility) == 2
    assert ranker.n_features_to_select == 10

def test_fit():
    ranker = Chi2()
    ranker.fit([[1, 2, 3], [4, 5, 6]], [0, 1])
    assert len(ranker.feature_importances_) == 3
    assert ranker.feature_importances_.sum() > 0