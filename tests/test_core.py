import hashlib

import numpy as np
import pandas as pd
import pytest

from gbd_mapping.id import reiid
from gbd_mapping.cause import causes
from gbd_mapping.risk import risk_factors

from vivarium_inputs import core


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


# TODO there's a bunch of repeated code in the next three functions but I'm not sure what the general form should be yet
@pytest.fixture
def mock_rrs(mocker):
    rrs_mock = mocker.patch("vivarium_inputs.core._get_relative_risk")
    rr_map = {}

    def rr_builder(risks, locations):
        rrs = []
        for risk in risks:
            for cause in risk.affected_causes:
                if (risk.gbd_id, cause.gbd_id) in rr_map:
                    current_rr = rr_map[(risk.gbd_id, cause.gbd_id)]
                else:
                    current_rr = int(hashlib.md5(str((risk.gbd_id, cause.gbd_id)).encode()).hexdigest(), 16)
                    current_rr /= 2**128
                    rr_map[(risk.gbd_id, cause.gbd_id)] = current_rr+1
                age_groups = [10, 11, 12]
                years = [1990, 1995]
                sexes = [1, 2]
                idx = pd.MultiIndex.from_product(
                    [[cause.gbd_id],   [risk.gbd_id], age_groups,     years,     sexes,    locations,  ["continuous"]],
                    names=["cause_id", "risk_id",     "age_group_id", "year_id", "sex_id", "location_id", "parameter"])
                rrs.append(pd.DataFrame({f"draw_{i}": current_rr for i in range(1000)}, index=idx).reset_index())
        return pd.concat(rrs)
    rrs_mock.side_effect = rr_builder
    return rrs_mock, rr_map


@pytest.fixture
def mock_pafs(mocker, cause_list):
    risk_cause_pafs = {}

    def mock_pafs(entity_ids, location_ids):
        pafs = []
        for gbd_id in entity_ids:
            if isinstance(gbd_id, reiid):
                rids = [gbd_id]
                # TODO This assumes that all non-cause entities are diarrhea etiologies
                cids = [causes.diarrheal_diseases.gbd_id]
            else:
                rids = {r.gbd_id for r in risk_factors if gbd_id in [cc.gbd_id for cc in r.affected_causes]}
                cids = [gbd_id]
            for c in cids:
                for r in rids:
                    if risk_cause_pafs:
                        if (r, c) in risk_cause_pafs:
                            current_paf = risk_cause_pafs[(r, c)]
                        else:
                            current_paf = int(hashlib.md5(str((r, c)).encode()).hexdigest(), 16)
                            current_paf /= 2**128
                            risk_cause_pafs[(r, c)] = current_paf
                    else:
                        current_paf = 0.001
                        risk_cause_pafs[(r, c)] = current_paf

                    age_groups = [10, 11, 12]
                    years = [1990, 1995]
                    sexes = [1, 2]
                    idx = pd.MultiIndex.from_product(
                        [[c],               [r],      age_groups,     years,     sexes,    [3],          location_ids],
                        names=["cause_id", "rei_id", "age_group_id", "year_id", "sex_id", "measure_id", "location_id"])
                    pafs.append(pd.DataFrame({f"draw_{i}":current_paf for i in range(1000)}, index=idx).reset_index())
        return pd.concat(pafs)

    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    gbd_mock.get_pafs.side_effect = mock_pafs
    return gbd_mock, risk_cause_pafs


@pytest.fixture
def mock_exposures(mocker):
    exposures_mock = mocker.patch("vivarium_inputs.core._get_exposure")
    exposure_map = {}

    def exposure_builder(risks, locations):
        exposures = []
        for risk in risks:
            if risk.gbd_id in exposure_map:
                current_exposure = exposure_map[risk.gbd_id]
            else:
                current_exposure = int(hashlib.md5(str(risk.gbd_id).encode()).hexdigest(), 16)
                current_exposure /= 2**128
                current_exposure *= 100
                exposure_map[risk.gbd_id] = current_exposure
            age_groups = [10, 11, 12]
            years = [1990, 1995]
            sexes = [1, 2]
            idx = pd.MultiIndex.from_product(
                [[risk.gbd_id],   age_groups,     years,     sexes,    locations,  ["continuous"]],
                names=["risk_id", "age_group_id", "year_id", "sex_id", "location_id", "parameter"])
            exposures.append(pd.DataFrame({f"draw_{i}": current_exposure
                                           for i in range(1000)}, index=idx).reset_index())
        return pd.concat(exposures)

    exposures_mock.side_effect = exposure_builder
    return exposures_mock, exposure_map


@pytest.mark.skip("Cluster")
def test__compute_paf_for_special_cases(mock_rrs, mock_exposures, locations):
    _, rrs = mock_rrs
    _, exposures = mock_exposures

    # TODO: This list is canonically specified as a constant inside _get_population_attributable_fraction
    # where it isn't really accessible for tests. Should probably clean that up.
    special_risks = [risk_factors.unsafe_water_source]

    location_ids = [core.get_location_ids_by_name()[name] for name in locations]
    for risk in special_risks:
        for cause in risk.affected_causes:
            paf = core._compute_paf_for_special_cases(cause, risk, location_ids)
            assert cause.gbd_id in paf.cause_id.values
            assert risk.gbd_id in paf.risk_id.values
            e = exposures[risk.gbd_id]
            rr = rrs[(risk.gbd_id, cause.gbd_id)]
            true_paf = (rr*e - 1) / (rr*e)
            assert all(paf[[f"draw_{i}" for i in range(1000)]] == true_paf)


@pytest.fixture(params=["cause", "etiology"])
def cause_like_entities(request, cause_list, etiology_list):
    if request.param == "cause":
        return cause_list
    elif request.param == "etiology":
        return etiology_list

@pytest.mark.skip("Cluster")
def test_get_draws_bad_args(cause_list, risk_list, locations):
    with pytest.raises(core.InvalidQueryError):
        core.get_draws(cause_list + risk_list, ['test'], locations)

    for measure in ['death', 'remission', 'prevalence', 'incidence', 'population_attributable_fraction']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(risk_list, [measure], locations)

    for measure in ['exposure', 'relative_risk']:
        with pytest.raises(core.InvalidQueryError):
            core.get_draws(cause_list, [measure], locations)


def test_get_relative_risk(mocker):
    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    gbd_mock.get_age_group_id.return_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    draw_cols = [f"rr_{i}" for i in range(10)]
    rr_maps = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2], 'age_group_id': [4, 5],
               'risk_id': [240], 'cause_id': [302], 'parameter': ['cat1', 'cat2', 'cat3', 'cat4'],
               'morbidity': [1], 'mortality': [1], 'rei_id ': [240], 'modelable_entity_id': [9082], 'metric_id': [3]}

    rr_ = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*rr_maps.values()], names=[*rr_maps.keys()]))
    rr_[draw_cols] = np.random.random_sample((len(rr_), 10)) * 10
    gbd_mock.get_relative_risks.return_value = rr_.reset_index()
    get_rr = core._get_relative_risk([risk_factors.child_wasting], [1])
    whole_age_groups = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    missing_age_groups = list(set(whole_age_groups) - set(rr_maps['age_group_id']))
    missing_rr_maps = rr_maps.copy()
    missing_rr_maps['age_group_id'] = missing_age_groups

    del missing_rr_maps['morbidity']
    del missing_rr_maps['mortality']
    del missing_rr_maps['metric_id']
    del missing_rr_maps['modelable_entity_id']

    missing_rr = pd.DataFrame(1.0, columns=[f"draw_{i}" for i in range(10)],
                              index=pd.MultiIndex.from_product([*missing_rr_maps.values()],
                              names=[*missing_rr_maps.keys()]))
    rr_ = rr_.rename(columns={f'rr_{i}': f'draw_{i}' for i in range(10)})
    rr_ = rr_.reset_index(['morbidity', 'mortality', 'metric_id', 'modelable_entity_id'])
    rr_ = rr_.drop(['morbidity', 'mortality', 'metric_id', 'modelable_entity_id'], axis=1)
    expected_rr = rr_.append(missing_rr).sort_index().reset_index()
    get_rr = get_rr[expected_rr.columns]

    pd.util.testing.assert_frame_equal(expected_rr, get_rr)


def test_get_population_attributable_fraction(mocker):
    id_mock = mocker.patch("vivarium_inputs.core._get_ids_for_measure")
    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    gbd_mock.get_age_group_id.return_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    draw_cols = [f"draw_{i}" for i in range(10)]
    paf_maps = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2], 'age_group_id': [4, 5],
                'rei_id': [84, 339, 238, 136, 240], 'cause_id': [302], 'measure_id':[3]}

    paf_ = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*paf_maps.values()], names=[*paf_maps.keys()]))
    paf_[draw_cols] = np.random.random_sample((len(paf_), 10))
    gbd_mock.get_pafs.return_value = paf_.reset_index()

    get_paf = core._get_population_attributable_fraction([causes.diarrheal_diseases], [1])

    whole_age_groups = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    missing_age_groups = list(set(whole_age_groups) - set(paf_maps['age_group_id']))
    missing_paf_maps = paf_maps.copy()
    missing_paf_maps['age_group_id'] = missing_age_groups
    del missing_paf_maps['measure_id']

    missing_paf = pd.DataFrame(0.0, columns=draw_cols,
                               index=pd.MultiIndex.from_product([*missing_paf_maps.values()],
                                                                names=[*missing_paf_maps.keys()]))
    paf_ = paf_.reset_index('measure_id')
    paf_ = paf_.drop('measure_id', axis=1)
    expected_paf = paf_.append(missing_paf).sort_index().reset_index()
    expected_paf = expected_paf.rename(columns={'rei_id':'risk_id'})
    get_paf = get_paf[expected_paf.columns]

    pd.util.testing.assert_frame_equal(expected_paf, get_paf)
