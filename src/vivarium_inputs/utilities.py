from typing import List
import pandas as pd

from core_maths.interpolate import pchip_interpolate
from vivarium_gbd_access import gbd


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


DEMOGRAPHIC_COLS = ('location_id', 'sex_id', 'age_group_id', 'year_id')
DRAW_COLS = tuple([f'draw_{i}' for i in range(1000)])
COLS = {
    'sequela':{
        'prevalence':
            ('measure_id', 'sequela_id', 'metric_id') + DRAW_COLS + DEMOGRAPHIC_COLS,
        'incidence':
            ('measure_id', 'sequela_id', 'metric_id') + DRAW_COLS + DEMOGRAPHIC_COLS,
        'disability_weight':
            ('location_id', 'age_group_id', 'sex_id', 'measure', 'healthstate_id', 'healthstate') + DRAW_COLS
    }
}


class VivariumInputsError(Exception):
    """Base exception for errors in vivarium_inputs."""
    pass


class DataNotExistError(VivariumInputsError, FileNotFoundError):
    """Exception raised when the gbd data for the entity-measure pair do not exist"""
    pass


class DataAbnormalError(VivariumInputsError, ValueError):
    """Exception raised when data has extra columns or values that we do not expect to have"""


def normalize(data: pd.DataFrame, location: int, fill_value=None) -> pd.DataFrame:
    data = normalize_location(data, location)
    data = normalize_sex(data)
    data = normalize_year(data)
    data = normalize_age(data, fill_value)
    keycols = list(set(['location_id', 'sex_id', 'age_group_id', 'year_id']).intersection(set(data.columns)))
    return data.sort_values(keycols).reset_index()


def reshape(data: pd.DataFrame, to_keep=DEMOGRAPHIC_COLS) -> pd.DataFrame:
    to_drop = set(data.columns) - set(DRAW_COLS) - set(to_keep)
    data.drop(to_drop, axis=1, inplace=True)
    data = pd.melt(data, id_vars=to_keep, value_vars=DRAW_COLS, var_name='draw')
    data["draw"] = data.draw.str.partition("_")[2].astype(int)
    return data


def remove_ids(data: pd.DataFrame) -> pd.DataFrame:
    location = data.location_id.unique()
    location_name = get_location_name(location[0])
    data['location'] = location_name
    data = data.rename(columns={'year_id': 'year'})
    data["sex"] = data.sex_id.map({1: "Male", 2: "Female", 3: "Both"})
    data.drop(["sex_id", "location_id"], axis=1,inplace=True)

    if 'age_group_id' in data:
        data = get_age_group_bins_from_age_group_id(data)
    return data


def normalize_location(data:pd.DataFrame, location: int)-> pd.DataFrame:
    location_id = data.location_id.unique()
    if len(location_id) != 1:
        raise DataAbnormalError(f'Data has extra location ids {set(location_id).difference({location})} '
                                f'other than {location}')
    elif location_id == [1]:
        data.loc[:, 'location_id'] = location
    elif location_id != [location]:
        raise DataAbnormalError(f'Data called for {location} has a location id {set(location_id)}')
    return data


def normalize_sex(data: pd.DataFrame) -> pd.DataFrame:
    sexes = data.sex_id.unique()
    if set(sexes) == {1, 2, 3}:
        data = data[data.sex_id.isin([1, 2])]
    elif set(sexes) == {3}:
        fill_data = data.copy()
        data.loc[:, 'sex_id'] = 1
        fill_data.loc[:, 'sex_id'] = 2
        data = pd.concat([data, fill_data])
    elif len(sexes) == 1:
        fill_data = data.copy()
        missing_sex = {1, 2}.difference(sexes)
        fill_data.loc[:, 'sex_id'] = missing_sex.pop()
        data = pd.concat([data, fill_data])
    return data


def normalize_year(data: pd.DataFrame) -> pd.DataFrame:
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    if 'year_id' not in data:
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df)
    else:
        year_ids = data.year_id.unique()
        id_cols = [c for c in data.columns if 'draw' not in c]
        value_cols = list(set(data.columns) - set(id_cols))
        if set(year_ids) == set(years['binned']):
            fillin_data = pchip_interpolate(data, id_cols, value_cols)
            data = pd.concat([data, fillin_data])
    return data[data.year_id.isin(years['annual'])]


def normalize_age(data: pd.DataFrame, fill_value) -> pd.DataFrame:
    if 'age_group_id' not in data:
        pass
    else: 
        whole_age_groups = gbd.get_age_group_id()
        age_groups = data.age_group_id.unique()
        if set(age_groups) == {22}:
            df = []
            for age in whole_age_groups:
                missing = data.copy()
                missing.loc[:, 'age_group_id'] = age
                df.append(missing)
            data = pd.concat(df, ignore_index=True)

        elif set(age_groups) < set(whole_age_groups):
            sex_id = data.sex_id.unique()
            year_id = data.year_id.unique()
            location_id = data.location_id.unique()

            index_cols = ['year_id', 'location_id', 'sex_id', 'age_group_id']
            draw_cols = [c for c in data.columns if 'draw_' in c]

            other_cols = {c: data[c].unique() for c in data.dropna(axis=1).columns if
                          c not in index_cols and c not in draw_cols}
            index_cols += [*other_cols.keys()]
            data = data.set_index(index_cols)

            expected = pd.MultiIndex.from_product([year_id, location_id, sex_id, whole_age_groups] + [*other_cols.values()],
                                                  names=(['year_id', 'location_id', 'sex_id', 'age_group_id'] + [
                                                      *other_cols.keys()]))
            new_data = data.copy()
            missing = expected.difference(data.index)
            to_add = pd.DataFrame({column: fill_value for column in draw_cols}, index=missing, dtype=float)
            new_data = new_data.append(to_add).sort_index()
            data = new_data.reset_index()

    return data


def get_location_name(location_id):
    return {r.location_id: r.location_name for _, r in gbd.get_location_ids().iterrows()}[location_id]


def get_age_group_bins_from_age_group_id(df):
    """Creates "age_group_start" and "age_group_end" columns from the "age_group_id" column

    Parameters
    ----------
    df: df for which you want an age column that has an age_group_id column

    Returns
    -------
    df with "age_group_start" and "age_group_end" columns
    """
    if df.empty:
        df['age_group_start'] = 0
        df['age_group_end'] = 0
        return df

    df = df.copy()
    idx = df.index
    mapping = gbd.get_age_bins()
    mapping = mapping.set_index('age_group_id')

    df = df.set_index('age_group_id')
    df[['age_group_start', 'age_group_end']] = mapping[['age_group_years_start', 'age_group_years_end']]

    df = df.set_index(idx)

    return df



