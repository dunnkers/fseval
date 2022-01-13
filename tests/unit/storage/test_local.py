import os
import tempfile
from pathlib import Path

import pytest

from fseval.storage.local import LocalStorage


@pytest.fixture
def storage() -> LocalStorage:
    """Create storage adapter mounted to a random temporary directory."""
    tmpdir = tempfile.mkdtemp()
    storage: LocalStorage = LocalStorage(load_dir=tmpdir, save_dir=tmpdir)

    return storage


def test_save_pickle(storage: LocalStorage):
    # create random object
    filename = "some_obj.pickle"
    some_obj: dict = {"a": 2}

    # save to pickle
    storage.save_pickle(filename, some_obj)

    # assert existance
    filepath = Path(storage.get_save_dir()) / filename
    assert os.path.isfile(filepath)
    assert os.stat(filepath).st_size > 0


def test_load_pickle(storage: LocalStorage):
    # create random object
    filename = "some_obj.pickle"
    some_obj: dict = {"a": 2}

    # save to pickle and load again
    storage.save_pickle(filename, some_obj)
    other_obj: dict = storage.restore_pickle(filename)

    # assert existance
    assert isinstance(other_obj, dict)
    assert list(other_obj.keys()) == ["a"]
    assert other_obj["a"] == 2


def test_non_existant_save(storage):
    """Error should be raised when trying to save to a non-existant directory."""
    with pytest.raises(FileNotFoundError):
        storage.save_dir = "non_existant_filepath"
        storage.save_pickle("some_file.pickle", {})


def test_non_existant_load(storage):
    """Loading should fail softly when file not found."""
    storage.load_dir = "non_existant_filepath"
    storage.restore_pickle("some_file.pickle")
