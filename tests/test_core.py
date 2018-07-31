import pytest
import hashlib

import pandas as pd

from gbd_mapping.id import reiid
from gbd_mapping.cause import Cause, causes
from gbd_mapping.risk import risks
from gbd_mapping.etiology import Etiology

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
                rids = {r.gbd_id for r in risks if gbd_id in [cc.gbd_id for cc in r.affected_causes]}
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
    special_risks = [risks.unsafe_water_source]

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


@pytest.mark.skip("cluster")
def test_get_population_attributable_fraction(mock_pafs, mock_rrs, mock_exposures, cause_like_entities, locations):
    _, risk_cause_pafs = mock_pafs

    pafs = core.get_draws(cause_like_entities, ["population_attributable_fraction"], locations)
    if isinstance(cause_like_entities[0], Etiology):
        id_column = "etiology_id"
        paired_id_column = "cause_id"
        expected_paired_entities = {causes.diarrheal_diseases.gbd_id}
    else:
        id_column = "cause_id"
        paired_id_column = "risk_id"
        expected_paired_entities = {r.gbd_id for r in risks if set(cause_like_entities).intersection(r.affected_causes)}
    assert {c.gbd_id for c in cause_like_entities
            if c not in [causes.all_causes, causes.tetanus]} == set(pafs[id_column].unique())

    assert expected_paired_entities == set(pafs[paired_id_column].unique())

    assert set(locations) == set(pafs.location.unique())

    for (r, c), v in risk_cause_pafs.items():
        assert all(pafs.query(f"{id_column} == @r and cause_id == @c")[[f"draw_{i}" for i in range(1000)]] == v)

    if isinstance(cause_like_entities[0], Cause):
        # TODO: This list is canonically specified as a constant inside _get_population_attributable_fraction
        # where it isn't really accessible for tests. Should probably clean that up.
        special_risks = [risks.unsafe_water_source]
        location_ids = [core.get_location_ids_by_name()[name] for name in locations]
        for risk in special_risks:
            for cause in risk.affected_causes:
                special = core._compute_paf_for_special_cases(cause, risk, location_ids)
                special['location'] = special.location_id.apply(core.get_location_names_by_id().get)
                del special['location_id']
                del special['measure_id']
                special = special.set_index(['age_group_id', 'year_id', 'sex_id', 'cause_id', 'location', 'risk_id'])
                assert all(
                    pafs.drop('measure', 'columns').set_index(
                        ['age_group_id', 'year_id', 'sex_id', 'cause_id', 'location', 'risk_id']
                    ).reindex(special.index) == special
                )


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
