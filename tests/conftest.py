"""Configuration of the test session."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Load the identifiers.org registry of pymetadata once before the tests.

    pymetadata downloads the registry into its cache at the first import of
    `pymetadata.identifiers.registry` when the cached file is missing or older
    than a day. With pytest-xdist every worker imports it at the same time and
    reads the file while another worker writes it, so the main process loads
    it before the workers start.
    """
    if not hasattr(config, "workerinput"):
        from pymetadata.identifiers import registry  # noqa: F401
