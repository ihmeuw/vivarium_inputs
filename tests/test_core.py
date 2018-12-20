import hashlib
from functools import partial

import numpy as np
import pandas as pd
import pytest

from gbd_mapping.id import reiid
from gbd_mapping.cause import causes
from gbd_mapping.risk import risk_factors
from gbd_mapping.coverage_gap import coverage_gaps

from vivarium_inputs import core, utilities


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
                        [[c],               [r],      age_groups,     years,     sexes,    [3, 4],        location_ids],
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


@pytest.fixture(params=["cause", "etiology"])
def cause_like_entities(request, cause_list, etiology_list):
    if request.param == "cause":
        return cause_list
    elif request.param == "etiology":
        return etiology_list


def test_get_relative_risk(mocker):
    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    gbd_mock_utilities = mocker.patch("vivarium_inputs.utilities.gbd")
    gbd_mock_utilities.get_age_group_id.return_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    draw_cols = [f"rr_{i}" for i in range(10)]
    rr_maps = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2], 'age_group_id': [4, 5],
               'rei_id': [240], 'cause_id': [302], 'parameter': ['cat1', 'cat2', 'cat3', 'cat4'],
               'morbidity': [1], 'mortality': [1], 'rei_id ': [240], 'modelable_entity_id': [9082], 'metric_id': [3]}

    rr_ = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*rr_maps.values()], names=[*rr_maps.keys()]))
    rr_[draw_cols] = np.random.random_sample((len(rr_), 10)) * 10
    gbd_mock.get_relative_risk.return_value = rr_.reset_index()
    get_rr = core.get_relative_risk(risk_factors.child_wasting, 1)
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
    expected_rr['affected_measure'] = 'incidence_rate' # every row has morbidity flag on
    get_rr = get_rr[expected_rr.columns]

    pd.util.testing.assert_frame_equal(expected_rr, get_rr)


def test_get_population_attributable_fraction(mocker):

    categorical_risk = [r for r in risk_factors if r.distribution in ['polytomous','dichotomous']]
    coverage_gap_list = [c for c in coverage_gaps]

    for risk in categorical_risk:
        with pytest.raises(core.InvalidQueryError):
            core.get_population_attributable_fraction(risk, 180)

    for cg in coverage_gap_list:
        with pytest.raises(core.InvalidQueryError):
            core.get_population_attributable_fraction(cg, 180)

    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    gbd_mock_utilities = mocker.patch("vivarium_inputs.utilities.gbd")
    gbd_mock_utilities.get_age_group_id.return_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    draw_cols = [f"draw_{i}" for i in range(10)]
    paf_maps = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2], 'age_group_id': [4, 5],
                'rei_id': [106], 'cause_id': [493, 495], 'measure_id': [3, 4]}

    paf_ = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*paf_maps.values()], names=[*paf_maps.keys()]))
    paf_[draw_cols] = np.random.random_sample((len(paf_), 10))
    gbd_mock.get_paf.return_value = paf_.reset_index()

    get_paf = core.get_population_attributable_fraction(risk_factors.high_total_cholesterol, 1).reset_index()

    whole_age_groups = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
    missing_age_groups = list(set(whole_age_groups) - set(paf_maps['age_group_id']))
    missing_paf_maps = paf_maps.copy()
    missing_paf_maps['age_group_id'] = missing_age_groups
    del missing_paf_maps['measure_id']

    missing_paf = pd.DataFrame(0.0, columns=draw_cols,
                               index=pd.MultiIndex.from_product([*missing_paf_maps.values(),
                                                                 ['incidence_rate', 'excess_mortality']],
                                                                names=[*missing_paf_maps.keys(), 'affected_measure'])).reset_index()
    paf_ = paf_.reset_index()
    paf_['affected_measure'] = paf_['measure_id'].apply(lambda x: 'incidence_rate' if x == 3 else 'excess_mortality')
    paf_ = paf_.drop('measure_id', 'columns')
    expected_paf = paf_.append(missing_paf).sort_index().reset_index(drop=True)
    get_paf = get_paf[expected_paf.columns]

    index_cols = ['age_group_id', 'cause_id', 'sex_id', 'year_id', 'affected_measure']
    pd.util.testing.assert_frame_equal(expected_paf.sort_values(index_cols).reset_index(drop=True),
                                       get_paf.sort_values(index_cols).reset_index(drop=True))



def mock_sequelae_disability_weights(sequelae, dw_data):
    """Returns a mock of the table of disability weights that vivarium_inputs.utilities.gbd.get_auxiliary_data returns.
     Used to patch that function call. Data is supplied to the function."""
    healthstate_id_tuples = list(map(lambda s: (s.healthstate.name, s.healthstate.gbd_id), sequelae))
    healthstates, healthstate_ids = list(zip(*healthstate_id_tuples))
    data = {'location_id': 1, 'sex_id': 3, 'age_group_id': 22, 'measure': 'disability_weight',
            'healthstate_id': healthstate_ids, 'healthstate': healthstates}

    sequelae = pd.DataFrame(data=data)
    combined = pd.concat([sequelae, dw_data], axis=1)

    return combined


def mock_sequelae_prevalence(sequela, __, prev_data):
    """Mocks sequela prevalence data, used to patch the call to get_prevalence from core.py as a side effect. Data is
    supplied to the function."""
    sequela_data = prev_data[sequela.name]
    draw_cols = [f"draw_{i}" for i in range(sequela_data.shape[1])]
    id_data = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2],
            'age_group_id': [4, 5], 'measure_id': [5], 'metric_id': [3]}
    index = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*id_data.values()], names=[*id_data.keys()])).index
    sequela_data.index = index
    sequela_data = sequela_data.reset_index()
    sequela_data['sequela_id'] = sequela.gbd_id

    return sequela_data


def make_test_disability_weight_data(sequelae, num_draws=10):
    """Mock data for sequelae prevalence and disability weights, as well as their weighted sum. This can be used as
    expected output for cause-level disability weights."""

    draw_cols = [f"draw_{i}" for i in range(num_draws)]
    data = {'year_id': [1990, 1995, 2000], 'location_id': [1], 'sex_id': [1, 2],
            'age_group_id': [4, 5], 'measure_id': [5], 'metric_id': [3]}
    prev_frame = pd.DataFrame(columns=draw_cols, index=pd.MultiIndex.from_product([*data.values()], names=[*data.keys()]))

    prev_data = {}  # dict is necessary based on the parameters of the fxn being mocked
    dw_data = []
    expected = None
    for s in sequelae:
        prev_draws = pd.DataFrame(data=np.random.random_sample((len(prev_frame), num_draws)), columns=draw_cols)
        dw_draws = pd.DataFrame(data=np.random.random_sample((1, num_draws)), columns=draw_cols)
        prev_data[s.name] = prev_draws
        dw_data.append(dw_draws)

        if expected is None:
            expected = pd.DataFrame(prev_draws.values * dw_draws.values)
        else:
            expected += pd.DataFrame(prev_draws.values * dw_draws.values)

    dw_data = pd.concat(dw_data, axis=0).reset_index(drop=True)
    expected.columns = [f'draw_{i}' for i in range(num_draws)]

    return prev_data, dw_data, expected


def test__get_sequela_disability_weights(mocker):
    """essentially a test of whether the aux data disability map is subset correctly."""
    cause = causes.ischemic_heart_disease

    prev_data, dw_data, expected = make_test_disability_weight_data(cause.sequelae)

    # mock disability weights
    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    mock_disability_weight_map = mock_sequelae_disability_weights(cause.sequelae, dw_data)
    gbd_mock.get_auxiliary_data.return_value = mock_disability_weight_map

    expected = mock_disability_weight_map.loc[mock_disability_weight_map.healthstate_id == cause.sequelae[0].healthstate.gbd_id]
    actual = core._get_sequela_disability_weights(cause.sequelae[0])

    assert 'sequela_id' in actual.columns, "We assume sequela_id is included with sequela-level disability weights"
    pd.testing.assert_frame_equal(expected, actual.drop('sequela_id', 'columns'))


def test_get_disability_weight_cause(mocker):
    """Test to ensure prevalence weighted sum of disability weights is the cause-level disability weight."""

    cause = causes.urticaria

    prev_data, dw_data, expected = make_test_disability_weight_data(cause.sequelae)

    # mock disability weights
    gbd_mock = mocker.patch("vivarium_inputs.core.gbd")
    mock_disability_weight_map = mock_sequelae_disability_weights(cause.sequelae, dw_data)
    gbd_mock.get_auxiliary_data.return_value = mock_disability_weight_map

    # mock prevalence
    prevalence_mock = mocker.patch("vivarium_inputs.core.get_prevalence")
    prevalence_mock.side_effect = partial(mock_sequelae_prevalence, prev_data=prev_data)

    mock_location_id = 1.0
    cause_disability_weight = core.get_disability_weight(cause, mock_location_id)

    draw_columns = [col for col in cause_disability_weight.columns if col.startswith('draw_')]

    pd.testing.assert_frame_equal(expected, cause_disability_weight[draw_columns])

