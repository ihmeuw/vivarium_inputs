import pytest

from gbd_mapping import causes, risk_factors, etiologies



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
