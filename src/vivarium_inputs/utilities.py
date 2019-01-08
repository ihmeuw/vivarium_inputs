"""Errors and utility functions for input processing."""
import pandas as pd

from gbd_mapping import causes
from .globals import gbd, DataAbnormalError, DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]

##################################################
# Functions to remove GBD conventions from data. #
##################################################


def scrub_gbd_conventions(data, location):
    data = scrub_location(data, location)
    data = scrub_sex(data)
    data = scrub_age(data)
    data = scrub_year(data)
    data = scrub_affected_entity(data)
    return data


def scrub_location(data, location):
    if 'location_id' in data.columns:
        data = data.drop('location_id', 'columns')
    data['location'] = location
    return data


def scrub_sex(data):
    if 'sex_id' in data.columns:
        data['sex'] = data['sex_id'].map({1: 'Male', 2: 'Female'})
        data = data.drop('sex_id', 'columns')
    return data


def scrub_age(data):
    if 'age_group_id' in data.columns:
        age_bins = (
            gbd.get_age_bins()[['age_group_id', 'age_group_years_start', 'age_group_years_end']]
                .rename(columns={'age_group_years_start': 'age_group_start',
                                 'age_group_years_end': 'age_group_end'})
        )
        data = data.merge(age_bins, on='age_group_id').drop('age_group_id', 'columns')
    return data


def scrub_year(data):
    if 'year_id' in data.columns:
        data = data.rename(columns={'year_id': 'year_start'})
        data['year_end'] = data['year_start'] + 1
    return data


def scrub_affected_entity(data):
    CAUSE_BY_ID = {c.gbd_id: c for c in causes}
    # RISK_BY_ID = {r.gbd_id: r for r in risk_factors}
    if 'cause_id' in data.columns:
        data['affected_entity'] = data.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
        data.drop('cause_id', axis=1, inplace=True)
    return data


###############################################################
# Functions to normalize GBD data over a standard demography. #
###############################################################


def normalize(data: pd.DataFrame, location_id: int, fill_value=None) -> pd.DataFrame:
    data = normalize_location(data, location_id)
    data = normalize_sex(data, fill_value)
    data = normalize_year(data)
    data = normalize_age(data, fill_value)
    return data


def normalize_location(data: pd.DataFrame, expected_location_id: int)-> pd.DataFrame:
    location_id = set(data.location_id.unique())
    if len(location_id) != 1:
        raise DataAbnormalError(f'Data has extra location ids {location_id.difference({expected_location_id})} '
                                f'other than {expected_location_id}')
    elif location_id != {expected_location_id}:
        raise DataAbnormalError(f'Data called for {expected_location_id} has a location id {location_id}')
    elif location_id == [1]:  # Make global data location specific
        data.loc[:, 'location_id'] = expected_location_id
    return data


def normalize_sex(data: pd.DataFrame, fill_value) -> pd.DataFrame:
    sexes = set(data.sex_id.unique())
    if sexes == {1, 2, 3}:  # We have variation across sex, don't need the column for both.
        data = data[data.sex_id.isin([1, 2])]
    elif sexes == {3}:  # Data is not sex specific, but does apply to both sexes, so copy.
        fill_data = data.copy()
        data.loc[:, 'sex_id'] = 1
        fill_data.loc[:, 'sex_id'] = 2
        data = pd.concat([data, fill_data], ignore_index=True)
    elif len(sexes) == 1:  # Data is sex specific, but only applies to one sex, so fill the other with default.
        fill_data = data.copy()
        missing_sex = {1, 2}.difference(sexes).pop()
        fill_data.loc[:, 'sex_id'] = missing_sex
        fill_data.loc[:, 'value'] = fill_value
        data = pd.concat([data, fill_data], ignore_index=True)
    return data


def normalize_year(data: pd.DataFrame) -> pd.DataFrame:
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    if 'year_id' not in data:  # Data doesn't vary by year, so copy for each year.
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df, ignore_index=True)
    elif set(data.year_id.unique()) == set(years['binned']):
        data = interpolate_year(data)
    elif set(data.year_id.unique()) < set(years['annual']):
        raise DataAbnormalError(f'No normalization scheme available for years {set(data.year_id.unique())}')

    return data[data.year_id.isin(years['annual'])]


def interpolate_year(data):
    # Hide the central comp dependency unless required.
    from core_maths.interpolate import pchip_interpolate
    id_cols = data.columns.difference(DRAW_COLUMNS)
    fillin_data = pchip_interpolate(data, id_cols, DRAW_COLUMNS)
    return pd.concat([data, fillin_data])


def normalize_age(data: pd.DataFrame, fill_value) -> pd.DataFrame:
    data_ages = set(data.age_group_id.unique())
    gbd_ages = set(gbd.get_age_group_id())

    if data_ages == {22}:  # Data applies to all ages, so copy.
        dfs = []
        for age in gbd_ages:
            missing = data.copy()
            missing.loc[:, 'age_group_id'] = age
            dfs.append(missing)
        data = pd.concat(dfs, ignore_index=True)
    elif data_ages < gbd_ages:
        key_columns = list(data.columns.difference(DRAW_COLUMNS))
        data = data.set_index(key_columns)
        key_columns.remove('age_group_id')

        expected_index = pd.MultiIndex.from_product([data[c].unique() for c in key_columns] + [gbd_ages],
                                                    names=key_columns + ['age_group_id'])
        data = data.reindex(expected_index, fill_value=fill_value)

    return data


def reshape(data: pd.DataFrame, to_keep=DEMOGRAPHIC_COLUMNS) -> pd.DataFrame:
    to_drop = set(data.columns) - set(DRAW_COLUMNS) - set(to_keep)
    data = data.drop(to_drop, 'columns')
    data = pd.melt(data, id_vars=to_keep, value_vars=DRAW_COLUMNS, var_name='draw')
    data["draw"] = data.draw.str.partition("_")[2].astype(int)
    return data


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    key_cols = []
    if 'draw' in data.columns:
        key_cols.append('draw')
    key_cols.extend([c for c in ['location', 'sex', 'age_group_start',
                                 'age_group_end', 'year_start', 'year_end'] if c in data.columns])
    other_cols = data.columns.difference(key_cols + ['value'])
    key_cols.extend(other_cols)
    data.sort_values(key_cols).reset_index(drop=True)

    data = data[key_cols + ['value']]
    return data
