import logging

import pytest
from _pytest.logging import caplog as _caplog
from gbd_mapping import causes, etiologies, risk_factors
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
def caplog(_caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message}")
    yield _caplog
    logger.remove(handler_id)


@pytest.fixture
def cause_list():
    return [
        causes.diarrheal_diseases,
        causes.ischemic_heart_disease,
        causes.ischemic_stroke,
        causes.hemorrhagic_stroke,
        causes.tetanus,
        causes.diabetes_mellitus,
        causes.all_causes,
    ]


@pytest.fixture
def etiology_list():
    return [etiologies.cholera, etiologies.amoebiasis]


@pytest.fixture
def sequela_list():
    return list(
        causes.diarrheal_diseases.sequelae
        + causes.ischemic_heart_disease.sequelae
        + causes.ischemic_stroke.sequelae
        + causes.hemorrhagic_stroke.sequelae
        + causes.hemorrhagic_stroke.sequelae
        + causes.tetanus.sequelae
        + causes.diabetes_mellitus.sequelae
    )


@pytest.fixture
def etiology_list():
    return list(
        causes.diarrheal_diseases.etiologies + causes.lower_respiratory_infections.etiologies
    )


@pytest.fixture
def risk_list():
    return [r for r in risk_factors]


@pytest.fixture
def locations():
    return ["Bangladesh", "Ethiopia", "Kenya", "China", "North Korea", "Nigeria"]
