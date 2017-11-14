import pytest

import pandas as pd
import numpy as np

from ceam_inputs.gbd_mapping import causes, risk_factors
from ceam_inputs import core

@pytest.fixture
def cause_list():
    return [causes.diarrheal_diseases, causes.ischemic_heart_disease, causes.ischemic_stroke,
            causes.hemorrhagic_stroke, causes.tetanus, causes.diabetes_mellitus, causes.all_causes]


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


def test_get_ids_for_deaths(cause_list, sequela_list):
    mapping = core.get_ids_for_measure(cause_list, ['deaths'])
    assert set(mapping['deaths']) == {c.gbd_id for c in cause_list}
    with pytest.raises(AssertionError):
        core.get_ids_for_measure(sequela_list, ['deaths'])


def test_get_ids_for_remission(cause_list, sequela_list):
    mapping = core.get_ids_for_measure(cause_list, ['remission'])
    assert set(mapping['remission']) == {c.dismod_id for c in cause_list}
    with pytest.raises(AssertionError):
        core.get_ids_for_measure(sequela_list, ['remission'])
    with pytest.raises(AssertionError):
        core.get_ids_for_measure([causes.age_related_and_other_hearing_loss], ['remission'])


def test_get_ids_for_prevalence(cause_list, sequela_list, etiology_list):
    mapping = core.get_ids_for_measure(cause_list, ['prevalence'])
    assert set(mapping['prevalence']) == {c.dismod_id for c in cause_list}
    mapping = core.get_ids_for_measure(cause_list, ['prevalence'])
    assert set(mapping['prevalence']) == {c.dismod_id for c in cause_list}
    with pytest.raises(AssertionError):
        core.get_ids_for_measure(sequela_list, ['remission'])
    with pytest.raises(AssertionError):
        core.get_ids_for_measure([causes.age_related_and_other_hearing_loss], ['remission'])


