"""Configuration of the test session."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Import the modules under test once in the main process before the tests.

    Some dependencies write into a shared cache when they are imported:
    pymetadata downloads the identifiers.org registry into its cache when the
    cached file is missing or older than a day, arviz writes a file for its
    daily warning. With pytest-xdist every worker imports them at the same
    time, one worker reads or renames the file while another one writes it and
    the collection fails. Importing them in the main process first leaves the
    workers a fresh cache.
    """
    if not hasattr(config, "workerinput"):
        import parvar.experiments.petab_factory  # noqa: F401
        import parvar.models.create_models  # noqa: F401
        import parvar.optimization.petab_optimization  # noqa: F401
