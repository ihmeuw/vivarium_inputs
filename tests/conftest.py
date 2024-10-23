from __future__ import annotations

import logging
from typing import Any

import pytest
from _pytest.logging import caplog as _caplog
from loguru import logger


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def runslow(request: pytest.FixtureRequest):
    """Fixture to allow 'runslow' to be used as an argument in tests."""
    return request.config.getoption("--runslow")


@pytest.fixture
def caplog(_caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message}")
    yield _caplog
    logger.remove(handler_id)


def pytest_report_teststatus(report: Any, config: Any) -> tuple[Any, str, str] | None:
    if report.when == "call" and report.skipped:
        if "todo" in report.longreprtext.lower():
            return (report.outcome, "T", "TODO")
        if "fixme" in report.longreprtext.lower():
            # NOTE: We mark the non-verbose status as "T" (the same as a TODO) because
            # "F" is reserved for failures.
            return (report.outcome, "T", "FIXME")


def _no_gbd_access():
    try:
        from vivarium_inputs.globals import GBDDummy

        return True
    except ImportError:
        pass
    return False


NO_GBD_ACCESS = _no_gbd_access()
