"""Errors and utility functions for input processing."""
import pandas as pd
import numpy as np
from typing import Union

from gbd_mapping import causes, risk_factors, Cause, RiskFactor
from vivarium_inputs.globals import gbd, DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS
from vivarium_inputs.validation.utilities import get_restriction_age_boundary, get_restriction_age_ids


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]


def get_age_bins():
    age_bins = (
        gbd.get_age_bins()[['age_group_id', 'age_group_name', 'age_group_years_start', 'age_group_years_end']]
            .rename(columns={'age_group_years_start': 'age_group_start',
                             'age_group_years_end': 'age_group_end'})
    )
    return age_bins


def get_annual_year_bins():
    estimation_years = gbd.get_estimation_years()
    df = pd.DataFrame({'year_id': range(min(estimation_years), max(estimation_years) + 1)})

    return scrub_year(df)


def get_demographic_dimensions(location_id, draws=False):
    ages = gbd.get_age_group_id()
    estimation_years = gbd.get_estimation_years()
    years = range(min(estimation_years), max(estimation_years) + 1)
    sexes = gbd.MALE + gbd.FEMALE
    location = [location_id]
    values = [location, sexes, ages, years]
    names = ['location_id', 'sex_id', 'age_group_id', 'year_id']
    if draws:
        values.append(range(1000))
        names.append('draw')

    dimensions = (pd.MultiIndex
                  .from_product(values, names=names)
                  .to_frame(index=False))

    return dimensions


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
        age_bins = get_age_bins()[['age_group_id', 'age_group_start', 'age_group_end']]
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


def normalize(data: pd.DataFrame, fill_value=None) -> pd.DataFrame:
    data = normalize_sex(data, fill_value)
    data = normalize_year(data)
    data = normalize_age(data, fill_value)
    return data


def normalize_sex(data: pd.DataFrame, fill_value) -> pd.DataFrame:
    sexes = set(data.sex_id.unique()) if 'sex_id' in data.columns else set()
    if not sexes:
        # Data does not correspond to individuals, so no age column necessary.
        pass
    elif sexes == {1, 2, 3}:
        # We have variation across sex, don't need the column for both.
        data = data[data.sex_id.isin([1, 2])]
    elif sexes == {3}:
        # Data is not sex specific, but does apply to both sexes, so copy.
        fill_data = data.copy()
        data.loc[:, 'sex_id'] = 1
        fill_data.loc[:, 'sex_id'] = 2
        data = pd.concat([data, fill_data], ignore_index=True)
    elif len(sexes) == 1:
        # Data is sex specific, but only applies to one sex, so fill the other with default.
        fill_data = data.copy()
        missing_sex = {1, 2}.difference(set(data.sex_id.unique())).pop()
        fill_data.loc[:, 'sex_id'] = missing_sex
        fill_data.loc[:, 'value'] = fill_value
        data = pd.concat([data, fill_data], ignore_index=True)
    else:  # sexes == {1, 2}
        pass
    return data


def normalize_year(data: pd.DataFrame) -> pd.DataFrame:
    binned_years = gbd.get_estimation_years()
    years = {'annual': list(range(min(binned_years), max(binned_years) + 1)), 'binned': binned_years}

    if 'year_id' not in data:
        # Data doesn't vary by year, so copy for each year.
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df, ignore_index=True)
    elif set(data.year_id) == set(years['binned']):
        data = interpolate_year(data)
    else:  # set(data.year_id.unique()) == years['annual']
        pass

    # Dump extra data.
    data = data[data.year_id.isin(years['annual'])]
    return data


def interpolate_year(data):
    # Hide the central comp dependency unless required.
    from core_maths.interpolate import pchip_interpolate
    id_cols = list(set(data.columns).difference(DRAW_COLUMNS))
    fillin_data = pchip_interpolate(data, id_cols, DRAW_COLUMNS)
    return pd.concat([data, fillin_data], sort=True)


def normalize_age(data: pd.DataFrame, fill_value) -> pd.DataFrame:
    data_ages = set(data.age_group_id.unique()) if 'age_group_id' in data.columns else set()
    gbd_ages = set(gbd.get_age_group_id())

    if not data_ages:
        # Data does not correspond to individuals, so no age column necessary.
        pass
    elif data_ages == {22}:
        # Data applies to all ages, so copy.
        dfs = []
        for age in gbd_ages:
            missing = data.copy()
            missing.loc[:, 'age_group_id'] = age
            dfs.append(missing)
        data = pd.concat(dfs, ignore_index=True)
    elif data_ages < gbd_ages:
        # Data applies to subset, so fill other ages with fill value.
        key_columns = list(data.columns.difference(DRAW_COLUMNS))
        key_columns.remove('age_group_id')
        expected_index = pd.MultiIndex.from_product([data[c].unique() for c in key_columns] + [gbd_ages],
                                                    names=key_columns + ['age_group_id'])

        data = (data.set_index(key_columns + ['age_group_id'])
                .reindex(expected_index, fill_value=fill_value)
                .reset_index())
    else:  # data_ages == gbd_ages
        pass
    return data


def reshape(data: pd.DataFrame, to_keep=DEMOGRAPHIC_COLUMNS) -> pd.DataFrame:
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
    data = data.sort_values(key_cols).reset_index(drop=True)

    sorted_cols = key_cols
    if 'value' in data.columns:
        sorted_cols += ['value']
    data = data[sorted_cols]
    return data


def convert_affected_entity(data: pd.DataFrame, column: str) -> pd.DataFrame:
    ids = data[column].unique()
    data = data.rename(columns={column: 'affected_entity'})
    if column == 'cause_id':
        name_map = {c.gbd_id: c.name for c in causes if c.gbd_id in ids}
    else:  # column == 'rei_id'
        name_map = {r.gbd_id: r.name for r in risk_factors if r.gbd_id in ids}
    data['affected_entity'] = data['affected_entity'].map(name_map)
    return data


def compute_categorical_paf(rr_data: pd.DataFrame, e: pd.DataFrame, affected_entity:str) -> pd.DataFrame:
    rr = rr_data[rr_data.affected_entity == affected_entity]
    affected_measure = rr.affected_measure.unique()[0]
    rr.drop(['affected_entity', 'affected_measure'], axis=1, inplace=True)

    key_cols = ['sex_id', 'age_group_id', 'year_id', 'parameter', 'draw']
    e = e.set_index(key_cols).sort_index(level=key_cols)
    rr = rr.set_index(key_cols).sort_index(level=key_cols)

    weighted_rr = e * rr
    groupby_cols = [c for c in key_cols if c != 'parameter']
    mean_rr = weighted_rr.reset_index().groupby(groupby_cols)['value'].sum()
    paf = ((mean_rr - 1) / mean_rr).reset_index()
    paf = paf.replace(-np.inf, 0)  # Rows with zero exposure.

    paf['affected_entity'] = affected_entity
    paf['affected_measure'] = affected_measure
    return paf


def get_age_group_ids_by_restriction(entity: Union[RiskFactor, Cause], which_age: str) -> (float,float):
    if which_age == 'yll':
        start, end = entity.restrictions.yll_age_group_id_start, entity.restrictions.yll_age_group_id_end
    elif which_age == 'yld':
        start, end = entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end
    elif which_age == 'inner':
        start = get_restriction_age_boundary(entity, 'start', reverse=True)
        end = get_restriction_age_boundary(entity, 'end', reverse=True)
    elif which_age == 'outer':
        start = get_restriction_age_boundary(entity, 'start')
        end = get_restriction_age_boundary(entity, 'end')
    else:
        raise NotImplementedError('The second argument of this function should be one of [yll, yld, inner, outer].')
    return start, end


def filter_data_by_restrictions(data: pd.DataFrame, entity: Union[RiskFactor, Cause], which_age: str) -> pd.DataFrame:
    """
    For the given data and restrictions, it applies age/sex restrictions and
    filter out the data outside of the range. Age restrictions can be applied
    in 4 different ways:
        - yld, yll, narrowest(inner) range of yll and yld,
        broadest(outer) range of yll and yld.

    Parameters
    ----------
    data
        DataFrame containing 'age_group_id' and 'sex_id' columns.
    entity
        Cause or RiskFactor
    which_age
        one of 4 choices: 'yll', 'yld', 'inner', 'outer'.

    Returns
    -------
        DataFrame which is filtered out any data outside of age/sex
        restriction ranges.
    """
    restrictions = entity.restrictions
    if restrictions.male_only and not restrictions.female_only:
        sexes = gbd.MALE
    elif not restrictions.male_only and restrictions.female_only:
        sexes = gbd.FEMALE
    else:  # not male only and not female only
        sexes = gbd.FEMALE + gbd.MALE + gbd.COMBINED

    data = data[data.sex_id.isin(sexes)]

    start, end = get_age_group_ids_by_restriction(entity, which_age)
    all_ages_id = 22
    ages = get_restriction_age_ids(start, end) + [all_ages_id]
    data = data[data.age_group_id.isin(ages)]
    return data
