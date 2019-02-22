"""Errors and utility functions for input processing."""
from numbers import Real
from typing import Union, List

from gbd_mapping import causes, risk_factors, Cause, RiskFactor
import numpy as np
import pandas as pd

from vivarium_inputs import utility_data
from vivarium_inputs.globals import DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS, SEXES, SPECIAL_AGES


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
        age_bins = (utility_data.get_age_bins()
                    .filter(['age_group_id', 'age_group_start', 'age_group_end'])
                    .set_index('age_group_id'))
        data['age_group_start'] = data['age_group_id'].map(age_bins['age_group_start'])
        data['age_group_end'] = data['age_group_id'].map(age_bins['age_group_end'])
        data = data.drop('age_group_id', 'columns')
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


def normalize(data: pd.DataFrame, fill_value: Real = None, cols_to_fill: List[str] = DRAW_COLUMNS) -> pd.DataFrame:
    data = normalize_sex(data, fill_value, cols_to_fill)
    data = normalize_year(data)
    data = normalize_age(data, fill_value, cols_to_fill)
    return data


def normalize_sex(data: pd.DataFrame, fill_value, cols_to_fill) -> pd.DataFrame:
    sexes = set(data.sex_id.unique()) if 'sex_id' in data.columns else set()
    if not sexes:
        # Data does not correspond to individuals, so no age column necessary.
        pass
    elif sexes == set(SEXES.values()):
        # We have variation across sex, don't need the column for both.
        data = data[data.sex_id.isin([SEXES['Male'], SEXES['Female']])]
    elif sexes == {SEXES['Combined']}:
        # Data is not sex specific, but does apply to both sexes, so copy.
        fill_data = data.copy()
        data.loc[:, 'sex_id'] = SEXES['Male']
        fill_data.loc[:, 'sex_id'] = SEXES['Female']
        data = pd.concat([data, fill_data], ignore_index=True)
    elif len(sexes) == 1:
        # Data is sex specific, but only applies to one sex, so fill the other with default.
        fill_data = data.copy()
        missing_sex = {SEXES['Male'], SEXES['Female']}.difference(set(data.sex_id.unique())).pop()
        fill_data.loc[:, 'sex_id'] = missing_sex
        fill_data.loc[:, cols_to_fill] = fill_value
        data = pd.concat([data, fill_data], ignore_index=True)
    else:  # sexes == {SEXES['Male'], SEXES['Female']}
        pass
    return data


def normalize_year(data: pd.DataFrame) -> pd.DataFrame:
    binned_years = utility_data.get_estimation_years()
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


def normalize_age(data: pd.DataFrame, fill_value: Real, cols_to_fill: List[str]) -> pd.DataFrame:
    data_ages = set(data.age_group_id.unique()) if 'age_group_id' in data.columns else set()
    gbd_ages = set(utility_data.get_age_group_ids())

    if not data_ages:
        # Data does not correspond to individuals, so no age column necessary.
        pass
    elif data_ages == {SPECIAL_AGES['all_ages']}:
        # Data applies to all ages, so copy.
        dfs = []
        for age in gbd_ages:
            missing = data.copy()
            missing.loc[:, 'age_group_id'] = age
            dfs.append(missing)
        data = pd.concat(dfs, ignore_index=True)
    elif data_ages < gbd_ages:
        # Data applies to subset, so fill other ages with fill value.
        key_columns = list(data.columns.difference(cols_to_fill))
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
    data = pd.melt(data.rename(columns={draw: i for i, draw in enumerate(DRAW_COLUMNS)}),
                   id_vars=to_keep, value_vars=range(len(DRAW_COLUMNS)), var_name='draw')
    data.draw = data.draw.astype(int)
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


def compute_categorical_paf(rr_data: pd.DataFrame, e: pd.DataFrame, affected_entity: str) -> pd.DataFrame:
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


def filter_data_by_restrictions(data: pd.DataFrame, entity: Union[RiskFactor, Cause],
                                which_age: str, age_group_ids: List[int]) -> pd.DataFrame:
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
    age_group_ids
        List of possible age group ids.

    Returns
    -------
        DataFrame which is filtered out any data outside of age/sex
        restriction ranges.
    """
    restrictions = entity.restrictions
    if restrictions.male_only and not restrictions.female_only:
        sexes = [SEXES['Male']]
    elif not restrictions.male_only and restrictions.female_only:
        sexes = [SEXES['Female']]
    else:  # not male only and not female only
        sexes = [SEXES['Male'], SEXES['Female'], SEXES['Combined']]

    data = data[data.sex_id.isin(sexes)]

    start, end = get_age_group_ids_by_restriction(entity, which_age)
    ages = get_restriction_age_ids(start, end, age_group_ids)
    data = data[data.age_group_id.isin(ages)]
    return data


def clear_disability_weight_outside_restrictions(data: pd.DataFrame, cause: Cause, fill_value: float,
                                                 age_group_ids: List[int]) -> pd.DataFrame:
    """Because sequela disability weight is not age/sex specific, we need to
    have a custom function to set the values outside the corresponding cause
    restrictions to 0 after it has been expanded over age/sex."""
    restrictions = cause.restrictions
    if restrictions.male_only and not restrictions.female_only:
        sexes = [SEXES['Male']]
    elif not restrictions.male_only and restrictions.female_only:
        sexes = [SEXES['Female']]
    else:  # not male only and not female only
        sexes = [SEXES['Male'], SEXES['Female'], SEXES['Combined']]

    start, end = get_age_group_ids_by_restriction(cause, "yld")
    ages = get_restriction_age_ids(start, end, age_group_ids)

    data.loc[(~data.sex_id.isin(sexes)) | (~data.age_group_id.isin(ages)), DRAW_COLUMNS] = fill_value
    return data


def filter_to_most_detailed_causes(data: pd.DataFrame)-> pd.DataFrame:
    """For the DataFrame including the cause_ids, it filters rows with
    cause_ids for the most detailed causes """
    cause_ids = set(data.cause_id)
    most_detailed_cause_ids = [c.gbd_id for c in causes if c.gbd_id in cause_ids and c.most_detailed]
    return data[data.cause_id.isin(most_detailed_cause_ids)]


def get_restriction_age_ids(start_id: Union[int, None], end_id: Union[int, None],
                            age_group_ids: List[int]) -> List[int]:
    """Get the start/end age group id and return the list of GBD age_group_ids
    in-between.
    """
    if start_id is None or end_id is None:
        data = []
    else:
        start_index = age_group_ids.index(start_id)
        end_index = age_group_ids.index(end_id)
        data = age_group_ids[start_index:end_index+1]
    return data


def get_restriction_age_boundary(entity: Union[RiskFactor, Cause], boundary: str, reverse=False):
    """Find the minimum/maximum age restriction (if both 'yll' and 'yld'
    restrictions exist) for a RiskFactor.

    Parameters
    ----------
    entity
        RiskFactor or Cause for which to find the minimum/maximum age restriction.
    boundary
        String 'start' or 'end' indicating whether to return the minimum(maximum)
        start age restriction or maximum(minimum) end age restriction.
    reverse
        if reverse is True, return the maximum of start age restriction
        and minimum of end age restriction.

    Returns
    -------
        The age group id corresponding to the minimum or maximum start or end
        age restriction, depending on `boundary`, if both 'yll' and 'yld'
        restrictions exist. Otherwise, returns whichever restriction exists.
    """
    yld_age = entity.restrictions[f'yld_age_group_id_{boundary}']
    yll_age = entity.restrictions[f'yld_age_group_id_{boundary}']
    if yld_age is None:
        age = yll_age
    elif yll_age is None:
        age = yld_age
    else:
        start_op = max if reverse else min
        end_op = min if reverse else max
        age = end_op(yld_age, yll_age) if boundary == 'start' else start_op(yld_age, yll_age)
    return age


def get_exposure_and_restriction_ages(exposure: pd.DataFrame, entity: RiskFactor) -> set:
    """Get the intersection of age groups found in exposure data and entity
    restriction age range. Used to filter other risk data where
    using just exposure age groups isn't sufficient because exposure at the
    point of extraction is pre-filtering by age restrictions.

    Parameters
    ----------
    exposure
        Exposure data for `entity`.
    entity
        Entity for which to find the intersecting exposure and restriction ages.

    Returns
    -------
    Set of age groups found in both the entity's exposure data and in the
    entity's age restrictions.
    """
    exposure_age_groups = set(exposure.age_group_id)
    start, end = get_age_group_ids_by_restriction(entity, 'outer')
    restriction_age_groups = get_restriction_age_ids(start, end, utility_data.get_age_group_ids())
    valid_age_groups = exposure_age_groups.intersection(restriction_age_groups)

    return valid_age_groups

