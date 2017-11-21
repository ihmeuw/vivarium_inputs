import pytest

from ceam_inputs.gbd_mapping import causes
from ceam_inputs import core







def test_get_ids_for_inconsistent_entities(cause_list, sequela_list):
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list + sequela_list, ['test'])


def test_get_ids_for_deaths(cause_list, sequela_list):
    mapping = core.get_ids_for_measure(cause_list, ['deaths'])
    assert set(mapping['deaths']) == {c.gbd_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(sequela_list, ['deaths'])


def test_get_ids_for_remission(cause_list, sequela_list):
    mapping = core.get_ids_for_measure(cause_list, ['remission'])
    assert set(mapping['remission']) == {c.dismod_id for c in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(sequela_list, ['remission'])
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure([causes.age_related_and_other_hearing_loss], ['remission'])


def test_get_ids_for_prevalence(cause_list, sequela_list, etiology_list):
    mapping = core.get_ids_for_measure(cause_list, ['prevalence'])
    assert set(mapping['prevalence']) == {c.dismod_id for c in cause_list}
    mapping = core.get_ids_for_measure(sequela_list, ['prevalence'])
    assert set(mapping['prevalence']) == {s.dismod_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(etiology_list, ['prevalence'])


def test_get_ids_for_incidence(cause_list, sequela_list, etiology_list):
    mapping = core.get_ids_for_measure(cause_list, ['incidence'])
    assert set(mapping['incidence']) == {c.dismod_id for c in cause_list}
    mapping = core.get_ids_for_measure(sequela_list, ['incidence'])
    assert set(mapping['incidence']) == {s.dismod_id for s in sequela_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(etiology_list, ['incidence'])


def test_get_ids_for_exposure(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['exposure'])
    assert set(mapping['exposure']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['exposure'])


def test_get_ids_for_rr(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['rr'])
    assert set(mapping['rr']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['rr'])


def test_get_ids_for_paf(cause_list, risk_list):
    mapping = core.get_ids_for_measure(risk_list, ['paf'])
    assert set(mapping['paf']) == {r.gbd_id for r in risk_list}
    with pytest.raises(core.InvalidQueryError):
        core.get_ids_for_measure(cause_list, ['rr'])


def test_get_gbd_draws_bad_args(cause_list, risk_list, locations):
    with pytest.raises(core.InvalidQueryError):
        core.get_gbd_draws(cause_list + risk_list, ['test'], locations)

    for measure in ['deaths', 'remission', 'prevalence', 'incidence']:
        with pytest.raises(core.InvalidQueryError):
            core.get_gbd_draws(risk_list, [measure], locations)

    for measure in ['exposure', 'rr', 'paf']:
        with pytest.raises(core.InvalidQueryError):
            core.get_gbd_draws(cause_list, [measure], locations)


def test_get_gbd_draws_deaths(cause_list, locations):








