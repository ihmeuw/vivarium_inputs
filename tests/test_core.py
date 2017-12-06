import pytest

from ceam_inputs.gbd_mapping import causes
from ceam_inputs import core


def test_get_ids_for_inconsistent_entities(cause_list, sequela_list):
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list + sequela_list, ['test'])


def test_get_ids_for_death(cause_list, sequela_list):
    mapping = core.get_ids_for_measure(cause_list, ['death'])
    assert set(mapping['death']) == {c.gbd_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(sequela_list, ['death'])


def test_get_ids_for_remission(sequela_list):
    cause_list = [causes.diarrheal_diseases, causes.tetanus, causes.diabetes_mellitus]
    mapping = core.get_ids_for_measure(cause_list, ['remission'])
    assert set(mapping['remission']) == {c.dismod_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(sequela_list, ['remission'])
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure([causes.age_related_and_other_hearing_loss], ['remission'])
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure([causes.ischemic_heart_disease], ['remission'])


def test_get_ids_for_prevalence(cause_list, sequela_list, etiology_list):
    mapping = core.get_ids_for_measure(cause_list, ['prevalence'])
    assert set(mapping['prevalence']) == {c.gbd_id for c in cause_list}
    mapping = core.get_ids_for_measure(sequela_list, ['prevalence'])
    assert set(mapping['prevalence']) == {s.gbd_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(etiology_list, ['prevalence'])


def test_get_ids_for_incidence(cause_list, sequela_list, etiology_list):
    mapping = core.get_ids_for_measure(cause_list, ['incidence'])
    assert set(mapping['incidence']) == {c.gbd_id for c in cause_list}
    mapping = core.get_ids_for_measure(sequela_list, ['incidence'])
    assert set(mapping['incidence']) == {s.gbd_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(etiology_list, ['incidence'])


def test_get_ids_for_exposure_mean(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['exposure_mean'])
    assert set(mapping['exposure_mean']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['exposure_mean'])


def test_get_ids_for_relative_risk(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['relative_risk'])
    assert set(mapping['relative_risk']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['relative_risk'])


def test_get_ids_for_population_attributable_fraction(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['population_attributable_fraction'])
    assert set(mapping['population_attributable_fraction']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['population_attributable_fraction'])


def test_get_gbd_draws_bad_args(cause_list, risk_list, locations):
    with pytest.raises(core.InvalidQueryError):
        core.get_gbd_draws(cause_list + risk_list, ['test'], locations)

    for measure in ['death', 'remission', 'prevalence', 'incidence']:
        with pytest.raises(core.InvalidQueryError):
            core.get_gbd_draws(risk_list, [measure], locations)

    for measure in ['exposure_mean', 'relative_risk', 'population_attributable_fraction']:
        with pytest.raises(core.InvalidQueryError):
            core.get_gbd_draws(cause_list, [measure], locations)
