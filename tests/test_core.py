import pytest

from ceam_inputs.gbd_mapping import causes, risk_factors
from ceam_inputs import core


def test_get_ids_for_inconsistent_entities(cause_list, sequela_list):
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(cause_list + sequela_list, 'test')


def test_get_ids_for_death(cause_list, sequela_list):
    ids = core._get_ids_for_measure(cause_list, 'death')
    assert set(ids) == {c.gbd_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(sequela_list, 'death')


def test_get_ids_for_remission(sequela_list):
    cause_list = [causes.diarrheal_diseases, causes.tetanus, causes.diabetes_mellitus]
    ids = core._get_ids_for_measure(cause_list, 'remission')
    assert set(ids) == {c.dismod_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(sequela_list, 'remission')
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure([causes.age_related_and_other_hearing_loss], 'remission')
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure([causes.ischemic_heart_disease], 'remission')


def test_get_ids_for_prevalence(cause_list, sequela_list, etiology_list):
    ids = core._get_ids_for_measure(cause_list, 'prevalence')
    assert set(ids) == {c.gbd_id for c in cause_list}
    ids = core._get_ids_for_measure(sequela_list, 'prevalence')
    assert set(ids) == {s.gbd_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(etiology_list, 'prevalence')


def test_get_ids_for_incidence(cause_list, sequela_list, etiology_list):
    ids = core._get_ids_for_measure(cause_list, 'incidence')
    assert set(ids) == {c.gbd_id for c in cause_list}
    ids = core._get_ids_for_measure(sequela_list, 'incidence')
    assert set(ids) == {s.gbd_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(etiology_list, 'incidence')


def test_get_ids_for_exposure(cause_list, risk_list):
    ids = core._get_ids_for_measure(risk_list, 'exposure')
    assert set(ids) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(cause_list, 'exposure')


def test_get_ids_for_relative_risk(cause_list, risk_list):
    ids = core._get_ids_for_measure(risk_list, 'relative_risk')
    assert set(ids) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(cause_list, 'relative_risk')


def test_get_ids_for_population_attributable_fraction(cause_list, risk_list):
    ids = core._get_ids_for_measure(risk_list, 'population_attributable_fraction')
    assert set(ids) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(cause_list, 'population_attributable_fraction')


def test_get_draws_bad_args(cause_list, risk_list, locations):
    with pytest.raises(core.InvalidQueryError):
        core.get_draws(cause_list + risk_list, ['test'], locations)

    for measure in ['death', 'remission', 'prevalence', 'incidence']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(risk_list, [measure], locations)

    for measure in ['exposure', 'relative_risk', 'population_attributable_fraction']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(cause_list, [measure], locations)

def test_get_draws__weird_risk_measures(locations):
    df = core.get_draws([risk_factors['high_systolic_blood_pressure']], ['exposure', 'relative_risk', 'population_attributable_fraction', 'exposure_standard_deviation'], [180])
