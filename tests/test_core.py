import pytest

import pandas as pd

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
    ids = core._get_ids_for_measure(cause_list, 'population_attributable_fraction')
    assert set(ids) == {r.gbd_id for r in cause_list}
    with pytest.raises(core.InvalidQueryError):
        core._get_ids_for_measure(risk_list, 'population_attributable_fraction')


def test_get_population_attributable_fraction(mocker, cause_list, locations):
    risk_cause_pafs = {}
    def mock_pafs(cause_ids, location_ids):
        pafs = []
        for c in cause_ids:
            for r in {r.gbd_id for r in risk_factors if set(cause_list).intersection(r.affected_causes)}:
                if risk_cause_pafs:
                    if (r, c) in risk_cause_pafs:
                        current_paf = risk_cause_pafs[(r, c)]
                    else:
                        current_paf = max(risk_cause_pafs.values()) + 0.001
                        risk_cause_pafs[(r, c)] = current_paf
                else:
                    current_paf = 0.001
                    risk_cause_pafs[(r, c)] = current_paf

                age_groups = [10, 11, 12]
                years = [1990, 1995]
                sexes = [1, 2]
                idx = pd.MultiIndex.from_product([[c],        [r],      age_groups,     years,     sexes,    [3],          location_ids],
                                           names=["cause_id", "rei_id", "age_group_id", "year_id", "sex_id", "measure_id", "location_id"])
                pafs.append(pd.DataFrame({f"draw_{i}":current_paf for i in range(1000)}, index=idx).reset_index())
        return pd.concat(pafs)


    gbd_mock = mocker.patch("ceam_inputs.core.gbd")
    gbd_mock.get_pafs.side_effect = mock_pafs

    pafs = core.get_draws(cause_list, ["population_attributable_fraction"], locations)
    assert {c.gbd_id for c in cause_list} == set(pafs.cause_id.unique())

    expected_risks = {r.gbd_id for r in risk_factors if set(cause_list).intersection(r.affected_causes)}
    assert expected_risks == set(pafs.risk_id.unique())

    assert set(locations) == set(pafs.location.unique())

    for (r, c), v in risk_cause_pafs.items():
        assert all(pafs.query("risk_id == @r and cause_id == @c")[[f"draw_{i}" for i in range(1000)]] == v)


def test_get_draws_bad_args(cause_list, risk_list, locations):
    with pytest.raises(core.InvalidQueryError):
        core.get_draws(cause_list + risk_list, ['test'], locations)

    for measure in ['death', 'remission', 'prevalence', 'incidence', 'population_attributable_fraction']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(risk_list, [measure], locations)

    for measure in ['exposure', 'relative_risk']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(cause_list, [measure], locations)


@pytest.mark.skip("This test has never passed?  Only relevant for data artifact.")
def test_get_draws__weird_risk_measures(locations):
    df = core.get_draws(
        [risk_factors['high_systolic_blood_pressure']],
        ['exposure', 'relative_risk', 'population_attributable_fraction', 'exposure_standard_deviation'],
        [180]
    )


