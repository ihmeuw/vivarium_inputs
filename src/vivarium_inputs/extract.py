import pandas as pd
import warnings
from typing import List

from vivarium_gbd_access import gbd
from gbd_mapping.sequela import Sequela

from .utilities import DataNotExistError, DataAbnormalError


DRAW_COLS = [f'draw_{i}' for i in range(1000)]
COLS = {
    'sequela':{
        'prevalence':
            ['year_id', 'age_group_id', 'sex_id', 'measure_id', 'sequela_id', 'location_id', 'metric_id'] + DRAW_COLS,
        'incidence':
            ['year_id', 'age_group_id', 'sex_id', 'measure_id', 'sequela_id', 'location_id', 'metric_id'] + DRAW_COLS,
        'disability_weight':
            ['location_id', 'age_group_id', 'sex_id', 'measure', 'healthstate_id', 'healthstate'] + DRAW_COLS
    }
}


def get_sequela_prevalence(entity: Sequela, location_id: int) -> pd.DataFrame:
    if not entity.prevalence_exists:
        raise DataNotExistError(f'{entity.name} does not have prevalence data')
    if not entity.prevalence_in_range:
        raise warnings.warn(f'{entity.name} has prevalence but its range is abnormal')
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type='sequela')
    data = data[data.measure_id == 5]

    assert data.metric_id.unique() == [3], 'prevalence should have only rate (metric_id 3)'

    expected_cols = COLS['sequela']['prevalence']
    check_columns(expected_cols, data.columns)
    check_years(data, 'annual')
    return data


def get_sequela_incidence(entity: Sequela, location_id: int) -> pd.DataFrame:
    if not entity.incidence_exists:
        raise DataNotExistError(f'{entity.name} does not have incidence data')
    if not entity.incidence_in_range:
        raise warnings.warn(f'{entity.name} has incidence but its range is abnormal')
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type='sequela')
    data = data[data.measure_id == 6]

    assert data.metric_id.unique() == [3], 'incidence should have only rate (metric_id 3)'
    expected_cols = COLS['sequela']['incidence']
    check_columns(expected_cols, data.columns)
    check_years(data, 'annual')
    return data


def get_sequela_birth_prevalence(entity: Sequela, location_id: int) -> pd.DataFrame:
    if not entity.birth_prevalence_exists:
        raise DataNotExistError(f'{entity.name} does not have incidence data')
    data = get_sequela_incidence(entity, location_id)
    return data[data.age_group_id == 164]


def get_sequela_disability_weight(entity: Sequela, _) -> pd.DataFrame:
    if not entity.healthstate.disability_weight_exist:
        raise DataNotExistError(f'{entity.name} does not have data for disability weight')
    disability_weights = gbd.get_auxiliary_data('disability_weight', 'sequela', 'all')
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    expected_cols = COLS['sequela']['disability_weight']
    check_columns(expected_cols, data.columns)
    return data


def check_years(df: pd.DataFrame, year_type: str):
    years = {'annual': [y for y in range(1990, 2018)], 'binned': gbd.get_estimation_years()}
    expected_years = years[year_type]
    if set(df.year_id.unique) < expected_years:
        raise DataNotExistError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique))}')


def check_columns(expected_cols:List, existing_cols:List):
    if set(existing_cols) < set(expected_cols):
        raise DataNotExistError(f'{set(expected_cols).difference(set(existing_cols))} columns are missing')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}')
