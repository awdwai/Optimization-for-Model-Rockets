import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "orhelper: needs orhelper + OpenRocket JAR + a .ork fixture"
    )
