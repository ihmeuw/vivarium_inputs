from itertools import product

import pytest
import pandas as pd
import numpy as np

from ceam_inputs.gbd_mapping import causes, risk_factors, sid
from ceam_inputs import core, gbd

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


@pytest.fixture
def locations():
    return [161, 179, 180, 6, 7, 214]


def cod_mock_output(cause_ids, location_ids):
    age = gbd.get_age_group_ids(gbd.GBD_ROUND_ID)
    measure = [1, 4]
    metric = [1]
    version = [66.]
    sex = [1, 2, 3]
    year = list(range(1980, 2017))
    idx_column_names = ['age_group_id', 'measure_id', 'metric_id', 'sex_id', 'year_id',
                        'cause_id', 'location_id', 'output_version_id']
    idx_column_values = zip(*product(age, measure, metric, sex, year, cause_ids, location_ids, version))
    cod_index = {name: values for name, values in zip(idx_column_names, idx_column_values)}
    cod_draws = {name: np.random.random(len(cod_index['age_group_id'])) for name in
                 [f'draw_{n}' for n in range(1000)]}

    df = pd.DataFrame(cod_index.update(cod_draws))
    df.loc[df.sex_id == 3, 'output_version_id'] = np.NaN

    return df


def me_mock_output(me_ids, location_ids):
    age = gbd.get_age_group_ids(gbd.GBD_ROUND_ID)
    measure = [5, 7, 9, 11, 12, 13, 14, 15, 16, 6]
    metric = [3]
    version = [190274.]
    sex = [1, 2, 3]
    year = [1990, 1995, 2000, 2005, 2010, 2016]
    idx_column_names = ['age_group_id', 'location_id', 'measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id', 'sex_id', 'year_id']
    idx_column_values = zip(*product(age, location_ids, measure, metric, version, me_ids, sex, year))

    me_index = {name: values for name, values in zip(idx_column_names, idx_column_values)}
    me_draws = {name: np.random.random(len(me_index['age_group_id'])) for name in
                 [f'draw_{n}' for n in range(1000)]}

    df = pd.DataFrame(me_index.update(me_draws))
    df.loc[df.sex_id == 3, 'model_version_id'] = np.NaN
    df.loc[df.sex_id == 3, 'modelable_entity_id'] = np.NaN

    return df


def como_mock_output(entity_ids, location_ids):
    age = gbd.get_age_group_ids(gbd.GBD_ROUND_ID)
    measure = [3, 5, 6]
    metric = [3]
    sex = [1, 2, 3]
    year = list(range(1980, 2017))

    id_col_name = 'sequela_id' if isinstance(entity_ids[0], sid) else 'cause_id'
    idx_column_names = ['age_group_id', id_col_name, 'location_id', 'measure_id', 'metric_id', 'sex_id', 'year_id']
    idx_column_values = zip(*product(age, entity_ids, location_ids, measure, metric, sex, year))

    como_index = {name: values for name, values in zip(idx_column_names, idx_column_values)}
    como_draws = {name: np.random.random(len(como_index['age_group_id'])) for name in
                  [f'draw_{n}' for n in range(1000)]}

    return pd.DataFrame(como_index.update(como_draws))


def rr_mock_output(risk_ids, location_ids):
    pass







@pytest.fixture
def gbd_mock(mocker):
    m = mocker.patch('ceam_inputs.core.gbd')
    m.get_codcorrect_draws.side_effect = cod_mock_output





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








