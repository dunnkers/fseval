from fseval.storage.mock import MockStorage


def test_mock_should_do_nothing():
    """MockStorage provider should have the following functions. All should do nothing.
    This Storage provider can be used in case one does not want to storage files at
    all."""
    mock: MockStorage = MockStorage()
    mock.get_load_dir()
    mock.get_save_dir()
    mock.save("", lambda: _)
    mock.save_pickle("", {})
    mock.restore("", lambda: _)
    mock.restore_pickle("")
