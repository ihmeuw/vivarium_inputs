import pandas as pd
import warnings
from typing import List

from vivarium_gbd_access import gbd
from gbd_mapping.sequela import Sequela

from .utilities import DataNotExistError, DataAbnormalError, METRIC, COLS


def get_sequela_prevalence(entity: Sequela, location_id: int) -> pd.DataFrame:
    if not entity.prevalence_exists:
        raise DataNotExistError(f'{entity.name} does not have prevalence data')
    if not entity.prevalence_in_range:
        raise warnings.warn(f'{entity.name} has prevalence but its range is abnormal')
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type='sequela')
    data = data[data.measure_id == 5]

    if data.metric_id.unique() != METRIC['Rate']:
        raise DataAbnormalError('prevalence should have only rate (metric_id 3)')

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

    if data.metric_id.unique() != METRIC['Rate']:
        raise DataAbnormalError('incidence should have only rate (metric_id 3)')
    expected_cols = COLS['sequela']['incidence']
    check_columns(expected_cols, data.columns)
    check_years(data, 'annual')
    return data


def get_sequela_disability_weight(entity: Sequela, _) -> pd.DataFrame:
    if not entity.healthstate.disability_weight_exist:
        raise DataNotExistError(f'{entity.name} does not have data for disability weight')
    disability_weights = gbd.get_auxiliary_data('disability_weight', 'sequela', 'all')
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    expected_cols = COLS['sequela']['disability_weight']
    check_columns(expected_cols, data.columns)
    return data


def check_years(df: pd.DataFrame, year_type: str):
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    expected_years = years[year_type]
    if set(df.year_id.unique()) < set(expected_years):
        raise DataAbnormalError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique()))}')
    # if is it annual, we expect to have extra years from some cases like codcorrect/covariate
    if year_type == 'binned' and set(df.year_id.unique()) > set(expected_years):
        raise DataAbnormalError(f'Data has extra years: {set(df.year_id.unique()).difference(set(expected_years))}')


def check_columns(expected_cols:List, existing_cols:List):
    if set(existing_cols) < set(expected_cols):
        raise DataNotExistError(f'{set(expected_cols).difference(set(existing_cols))} columns are missing')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}')
