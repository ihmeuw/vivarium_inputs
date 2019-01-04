import pandas as pd
import warnings
from typing import List

from vivarium_gbd_access import gbd
from gbd_mapping.sequela import Sequela

from .utilities import DataNotExistError, DataAbnormalError


METRIC = {
    'Number': 1,
    'Percent': 2,
    'Rate': 3,
    'Rank': 4,
    'Years': 5,
    'p-value': 6,
    'MDG p-value': 7,
    'Probability of death': 8,
    'Index score': 9
}

MEASURE = {
    'Deaths': 1, 'DALYs (Disability-Adjusted Life Years)': 2, 'YLDs (Years Lived with Disability)': 3,
    'YLLs (Years of Life Lost)': 4, 'Prevalence': 5, 'Incidence': 6, 'Remission': 7, 'Duration': 8,
    'Excess mortality rate': 9, 'Prevalence * excess mortality rate': 10, 'Relative risk': 11,
    'Standardized mortality ratio': 12, 'With-condition mortality rate': 13, 'All-cause mortality rate': 14,
    'Cause-specific mortality rate': 15, 'Other cause mortality rate': 16, 'Case fatality rate': 17, 'Proportion': 18,
    'Continuous': 19, 'Survival Rate': 20, 'Disability Weight': 21, 'Chronic Prevalence': 22, 'Acute Prevalence': 23,
    'Acute Incidence': 24, 'Maternal mortality ratio': 25, 'Life expectancy': 26, 'Probability of death': 27,
    'HALE (Healthy life expectancy)': 28, 'Summary exposure value': 29, 'Life expectancy no-shock hiv free': 30,
    'Life expectancy no-shock with hiv': 31, 'Probability of death no-shock hiv free': 32,
    'Probability of death no-shock with hiv': 33, 'Mortality risk': 34, 'Short term prevalence': 35,
    'Long term prevalence': 36, 'Life expectancy decomposition by cause': 37, 'Birth prevalence': 38,
    'Susceptible population fraction': 39, 'With Condition population fraction': 40, 'Susceptible incidence': 41,
    'Total incidence': 42, 'HAQ Index (Healthcare Access and Quality Index)': 43, 'Population': 44, 'Fertility': 45
}


DEMOGRAPHIC_COLS = ['location_id', 'sex_id', 'age_group_id', 'year_id']
DRAW_COLS = [f'draw_{i}' for i in range(1000)]
COLS = {
    'sequela':{
        'prevalence':
            ['measure_id', 'sequela_id', 'metric_id'] + DRAW_COLS + DEMOGRAPHIC_COLS,
        'incidence':
            ['measure_id', 'sequela_id', 'metric_id'] + DRAW_COLS + DEMOGRAPHIC_COLS,
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
        raise DataNotExistError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique()))}')
    # if is it annual, we expect to have extra years from some cases like codcorrect/covariate
    if year_type == 'binned' and set(df.year_id.unique()) > set(expected_years):
        raise DataAbnormalError(f'Data has extra years: {set(df.year_id.unique()).difference(set(expected_years))}')


def check_columns(expected_cols:List, existing_cols:List):
    if set(existing_cols) < set(expected_cols):
        raise DataNotExistError(f'{set(expected_cols).difference(set(existing_cols))} columns are missing')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}')
