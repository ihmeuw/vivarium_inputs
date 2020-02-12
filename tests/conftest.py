import pytest
import logging
from _pytest.logging import caplog as _caplog
from loguru import logger

from gbd_mapping import causes, risk_factors, etiologies


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
    return [causes.diarrheal_diseases, causes.ischemic_heart_disease, causes.ischemic_stroke,
            causes.hemorrhagic_stroke, causes.tetanus, causes.diabetes_mellitus, causes.all_causes]


@pytest.fixture
def etiology_list():
    return [etiologies.cholera, etiologies.amoebiasis]


@pytest.fixture
def sequela_list():
    return list(causes.diarrheal_diseases.sequelae + causes.ischemic_heart_disease.sequelae
                + causes.ischemic_stroke.sequelae + causes.hemorrhagic_stroke.sequelae
                + causes.hemorrhagic_stroke.sequelae + causes.tetanus.sequelae
                + causes.diabetes_mellitus.sequelae)


@pytest.fixture
def etiology_list():
    return list(causes.diarrheal_diseases.etiologies + causes.lower_respiratory_infections.etiologies)


@pytest.fixture
def risk_list():
    return [r for r in risk_factors]


@pytest.fixture
def locations():
    return ['Bangladesh', 'Ethiopia', 'Kenya', 'China', 'North Korea', 'Nigeria']
