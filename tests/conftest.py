def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dependency: registers pytest.mark.dependency "
        + "from `pytest-depedency` module",
    )
