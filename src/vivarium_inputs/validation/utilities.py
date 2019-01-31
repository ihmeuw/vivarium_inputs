import operator
from typing import Union, List
import warnings

import pandas as pd
import numpy as np

from vivarium_inputs.globals import (DRAW_COLUMNS, METRICS, MEASURES,
                                     DataAbnormalError, DataNotExistError, VivariumInputsError,
                                     gbd)
from gbd_mapping import RiskFactor, Cause


def check_years(data: pd.DataFrame, year_type: str, error: bool = True):
    """Check that years in passed data match expected range based on type.

    Parameters
    ----------
    data
        Dataframe containing 'year_id' column.
    year_type
        String 'annual' or 'binned' indicating expected year range.
    error
        Boolean indicating whether to raise an error if expected years are not
        found.

    Returns
    -------
    bool
        True if years in `data` match expected `year_type`, false otherwise.

    Raises
    ------
    DataAbnormalError
        If `error` is turned on and any expected years are not found in data or
        any extra years found and `year_type` is 'binned'.

    """
    gbd_years = gbd.get_estimation_years()
    years = {'annual': list(range(min(gbd_years), max(gbd_years)+1)), 'binned': gbd_years}
    expected_years = years[year_type]
    if set(data.year_id.unique()) < set(expected_years):
        if error:
            raise DataAbnormalError(f'Data has missing years: {set(expected_years).difference(set(data.year_id))}.')
        return False
    # if it's annual, we expect to have extra years from some sources (e.g., codcorrect/covariate)
    if year_type == 'binned' and set(data.year_id.unique()) > set(expected_years):
        if error:
            raise DataAbnormalError(f'Data has extra years: {set(data.year_id).difference(set(expected_years))}.')
        return False
    return True


def check_location(data: pd.DataFrame, location_id: int):
    """Check that data contains only a single unique location id and that that
    location id matches either the global location id or the
    requested `location_id`.

    Parameters
    ----------
    data
        Dataframe containing a 'location_id' column.
    location_id
        The requested location_id.

    Raises
    ------
    DataAbnormalError
        If data contains multiple location ids or a location id other than the
        global or requested location id.

    """
    if len(data['location_id'].unique()) > 1:
        raise DataAbnormalError(f'Data contains multiple location ids.')

    data_location_id = data['location_id'].unique()[0]

    location_metadata = gbd.get_location_metadata()
    path_to_parent = location_metadata.loc[location_metadata.location_id == location_id, 'path_to_top_parent'].max().split(',')
    path_to_parent = [int(i) for i in path_to_parent]

    if data_location_id not in path_to_parent:  
        raise DataAbnormalError(f'Data pulled for {location_id} actually has location id {data_location_id}, which is not not in its hiararchy.')

    if data_location_id != location_id:
        data['location_id'] = location_id


def check_columns(expected_cols: List, existing_cols: List):
    """Verify that the passed lists of columns match.

    Parameters
    ----------
    expected_cols
        List of column names expected.
    existing_cols
        List of column names actually found in data.

    Raises
    ------
    DataAbnormalError
        If `expected_cols` does not match `existing_cols`.

    """
    if set(existing_cols) < set(expected_cols):
        raise DataAbnormalError(f'Data is missing columns: {set(expected_cols).difference(set(existing_cols))}.')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}.')


def check_data_exist(data: pd.DataFrame, zeros_missing: bool,
                     value_columns: list = DRAW_COLUMNS, error: bool = True) -> bool:
    """Check that values in data exist and none are missing and, if
    `zeros_missing` is turned on, not all zero.

    Parameters
    ----------
    data
        Dataframe containing `value_columns`.
    zeros_missing
        Boolean indicating whether to treat all zeros in `value_columns` as
        missing or not.
    value_columns
        List of columns in `data` to check for missing values.
    error
        Boolean indicating whether or not to error if data is missing.

    Returns
    -------
    bool
        True if non-missing, non-infinite, non-zero (if zeros_missing) values
        exist in data, False otherwise.

    Raises
    -------
    DataNotExistError
        If error flag is set to true and data is empty or contains any NaN
        values in `value_columns`, or contains all zeros in `value_columns` and
        zeros_missing is True.

    """
    if (data.empty or np.any(pd.isnull(data[value_columns]))
            or (zeros_missing and np.all(data[value_columns] == 0)) or np.any(np.isinf(data[value_columns]))):
        if error:
            raise DataNotExistError(f'Data contains no non-missing{", non-zero" if zeros_missing else ""} values.')
        return False
    return True


def get_restriction_age_ids(start_id: Union[float, None], end_id: Union[float, None]) -> list:
    """Get the start/end age group id and return the list of GBD age_group_ids
    in-between.
    """
    if start_id is None:
        return []

    gbd_age_ids = gbd.get_age_group_id()
    start_index = gbd_age_ids.index(start_id)
    end_index = gbd_age_ids.index(end_id)

    return gbd_age_ids[start_index:end_index+1]


def _check_continuity(data_ages: set, all_ages: set):
    """Make sure data_ages is contiguous block in all_ages."""
    data_ages = list(data_ages)
    all_ages = list(all_ages)
    all_ages.sort()
    data_ages.sort()
    if all_ages[all_ages.index(data_ages[0]):all_ages.index(data_ages[-1])+1] != data_ages:
        raise DataAbnormalError(f'Data contains a non-contiguous age groups: {data_ages}.')


def check_age_group_ids(data: pd.DataFrame, restriction_start: float = None, restriction_end: float = None):
    """Check the set of age_group_ids included in data pulled from GBD for
    the following conditions:

        - if data ages contain invalid age group ids, error.
        - if data ages are equal to the set of all GBD age groups or the set of
        age groups within restriction bounds (if restrictions apply), pass.
        - if data ages are not a contiguous block of GBD age groups, error.
        - if data ages are a proper subset of the set of restriction age groups
        or the restriction age groups are a proper subset of the data ages,
        warn.


    Parameters
    ----------
    data
        Dataframe pulled containing age_group_id column.
    restriction_start
        Age group id representing the start of the restriction range
        if applicable.
    restriction_end
        Age group id representing the end of the restriction range
        if applicable.

    Raises
    ------
    DataAbnormalError
        If age group ids contained in data aren't all valid GBD age group ids
        or they don't make up a contiguous block.

    """
    all_ages = set(gbd.get_age_group_id())
    restriction_ages = set(get_restriction_age_ids(restriction_start, restriction_end))
    data_ages = set(data.age_group_id)

    invalid_ages = data_ages.difference(all_ages)
    if invalid_ages:
        raise DataAbnormalError(f'Data contains invalid age group ids: {invalid_ages}.')

    _check_continuity(data_ages, all_ages)

    if data_ages < restriction_ages:
        warnings.warn('Data does not contain all age groups in restriction range.')
    elif restriction_ages and restriction_ages < data_ages:
        warnings.warn('Data contains additional age groups beyond those specified by restriction range.')
    else:  # data_ages == restriction_ages
        pass


def check_sex_ids(data: pd.DataFrame, male_expected: bool = True, female_expected: bool = True,
                  combined_expected: bool = False):
    """Check whether the data contains valid GBD sex ids and whether the set of
    sex ids in the data matches the expected set.

    Parameters
    ----------
    data
        Dataframe containing a sex_id column.
    male_expected
        Boolean indicating whether the male sex id is expected in this data.
        For some data pulling tools, this may correspond to whether the entity
        the data describes has a male_only sex restriction.
    female_expected
        Boolean indicating whether the female sex id is expected in this data.
        For some data pulling tools, this may correspond to whether the entity
        the data describes has a female_only sex restriction.
    combined_expected
        Boolean indicating whether data is expected to include the
        combined sex id.

    Raises
    ------
    DataAbnormalError
        If data contains any sex ids that aren't valid GBD sex ids.

    """
    valid_sex_ids = gbd.MALE + gbd.FEMALE + gbd.COMBINED  # these are single-item lists
    gbd_sex_ids = set(np.array(valid_sex_ids)[[male_expected, female_expected, combined_expected]])
    data_sex_ids = set(data.sex_id)

    invalid_sex_ids = data_sex_ids.difference(set(valid_sex_ids))
    if invalid_sex_ids:
        raise DataAbnormalError(f'Data contains invalid sex ids: {invalid_sex_ids}.')

    extra_sex_ids = data_sex_ids.difference(gbd_sex_ids)
    if extra_sex_ids:
        warnings.warn(f'Data contains the following extra sex ids {extra_sex_ids}.')

    missing_sex_ids = set(gbd_sex_ids).difference(data_sex_ids)
    if missing_sex_ids:
        warnings.warn(f'Data is missing the following expected sex ids: {missing_sex_ids}.')


def check_age_restrictions(data: pd.DataFrame, age_group_id_start: int, age_group_id_end: int,
                           value_columns: list = DRAW_COLUMNS):
    """Check that all expected age groups between age_group_id_start and
    age_group_id_end, inclusive, and only those age groups, appear in data with
    non-missing values in `value_columns`.

    Parameters
    ----------
    data
        Dataframe containing an age_group_id column.
    age_group_id_start
        Lower boundary of age group ids expected in data, inclusive.
    age_group_id_end
        Upper boundary of age group ids expected in data, exclusive.
    value_columns
        List of columns to verify values are non-missing for expected age
        groups and missing for not expected age groups.

    Raises
    ------
    DataAbnormalError
        If any age group ids in the range
        [`age_group_id_start`, `age_group_id_end`] don't appear in the data or
        if any additional age group ids (with the exception of 235) appear in
        the data.

    """
    expected_gbd_age_ids = get_restriction_age_ids(age_group_id_start, age_group_id_end)

    # age groups we expected in data but that are not
    missing_age_groups = set(expected_gbd_age_ids).difference(set(data.age_group_id))
    extra_age_groups = set(data.age_group_id).difference(set(expected_gbd_age_ids))

    if missing_age_groups:
        raise DataAbnormalError(f'Data was expected to contain all age groups between ids '
                                f'{age_group_id_start} and {age_group_id_end}, '
                                f'but was missing the following: {missing_age_groups}.')
    if extra_age_groups:
        # we treat all 0s as missing in accordance with gbd so if extra age groups have all 0 data, that's fine
        should_be_zero = data[data.age_group_id.isin(extra_age_groups)]
        if check_data_exist(should_be_zero, zeros_missing=True, value_columns=value_columns, error=False):
            warnings.warn(f'Data was only expected to contain values for age groups between ids '
                          f'{age_group_id_start} and {age_group_id_end} (with the possible addition of 235), '
                          f'but also included values for age groups {extra_age_groups}.')

    # make sure we're not missing data for all ages in restrictions
    if not check_data_exist(data[data.age_group_id.isin(expected_gbd_age_ids)], zeros_missing=True,
                            value_columns=value_columns, error=False):
        raise DataAbnormalError(f'Data is missing for all age groups within restriction range.')


def check_value_columns_boundary(data: pd.DataFrame, boundary_value: Union[float, pd.Series], boundary_type: str,
                                 value_columns: list = DRAW_COLUMNS, inclusive: bool = True,
                                 error: type(VivariumInputsError) = None):
    """Check that all values in DRAW_COLUMNS in data are above or below given
    boundary_value.

    Parameters
    ----------
    data
        Dataframe containing `value_columns`.
    boundary_value
        Value against which `value_columns` values will be checked. May be a
        series of values with a matching index to data.
    boundary_type
        String 'upper' or 'lower' indicating whether `boundary_value` is upper
        or lower limit on `value_columns`.
    value_columns
        List of column names in `data`, the values of which should be checked
        against `boundary_value`.
    inclusive
        Boolean indicating whether `boundary_value` is inclusive or not.
    error
        Exception class indicating what error should be raised if values are
        found outside `boundary_value`. If none, warn instead of raising error.

    Raises
    -------
    DataAbnormalError
        If any values in `value_columns` are above/below `boundary_value`,
        depending on `boundary_type`, if `error` is turned on.
    """
    msg = f'Data contains values {"below" if boundary_type == "lower" else "above"} ' \
        f'{"or equal to " if not inclusive else ""}the expected boundary ' \
        f'value{"s" if isinstance(boundary_value, pd.Series) else f" ({boundary_value})"}.'

    if boundary_type == "lower":
        op = operator.ge if inclusive else operator.gt
        data_values = data[value_columns].min(axis=1)
    else:
        op = operator.le if inclusive else operator.lt
        data_values = data[value_columns].max(axis=1)

    if isinstance(boundary_value, pd.Series):
        data_values = data_values.sort_index()
        boundary_value = boundary_value.sort_index()

    if not np.all(op(data_values, boundary_value)):
        if error is not None:
            raise error(msg)
        else:
            warnings.warn(msg)


def check_sex_restrictions(data: pd.DataFrame, male_only: bool, female_only: bool,
                           value_columns: list = DRAW_COLUMNS):
    """Check that all expected sex ids based on restrictions, and only those
    sex ids, appear in data with non-missing values in `value_columns`.

    Parameters
    ----------
    data
        Dataframe contained sex_id column.
    male_only
        Boolean indicating whether data is restricted to male only estimates.
    female_only
        Boolean indicating whether data is restricted to female only estimates.
    value_columns
        List of columns to verify values are non-missing for expected sex
        ids and missing for not expected sex ids.

    Raises
    -------
    DataAbnormalError
        If data violates passed sex restrictions.
    """
    female, male, combined = gbd.FEMALE[0], gbd.MALE[0], gbd.COMBINED[0]

    if male_only:
        if not check_data_exist(data[data.sex_id == male], zeros_missing=True,
                                value_columns=value_columns, error=False):
            raise DataAbnormalError('Data is restricted to male only, but is missing data values for males.')

        if (set(data.sex_id) != {male} and
                check_data_exist(data[data.sex_id != male], zeros_missing=True,
                                 value_columns=value_columns, error=False)):
           warnings.warn('Data is restricted to male only, but contains '
                         'non-male sex ids for which data values are not all 0.')

    if female_only:
        if not check_data_exist(data[data.sex_id == female], zeros_missing=True,
                                value_columns=value_columns, error=False):
            raise DataAbnormalError('Data is restricted to female only, but is missing data values for females.')

        if (set(data.sex_id) != {female} and
                check_data_exist(data[data.sex_id != female], zeros_missing=True, 
                                 value_columns=value_columns, error=False)):
            warnings.warn('Data is restricted to female only, but contains '
                          'non-female sex ids for which data values are not all 0.')

    if not male_only and not female_only:
        if {male, female}.issubset(set(data.sex_id)):
            if (not check_data_exist(data[data.sex_id == male], zeros_missing=True,
                                     value_columns=value_columns, error=False) or
               not check_data_exist(data[data.sex_id == female], zeros_missing=True,
                                    value_columns=value_columns, error=False)):
                raise DataAbnormalError('Data has no sex restrictions, but does not contain non-zero '
                                        'values for both males and females.')
        else:  # check combined sex id
            if not check_data_exist(data[data.sex_id == combined], zeros_missing=True,
                                    value_columns=value_columns, error=False):
                raise DataAbnormalError('Data has no sex restrictions, but does not contain non-zero '
                                        'values for both males and females.')


def check_measure_id(data: pd.DataFrame, allowable_measures: List[str], single_only: bool = True):
    """Check that data contains a measure id that is one of the allowed
    measure ids.

    Parameters
    ----------
    data
        Dataframe containing 'measure_id' column.
    allowable_measures
        List of strings dictating the possible values for measure id when
        mapped via MEASURES.
    single_only
        Boolean indicating whether a single measure id is expected in the data
        or whether multiple are allowable.

    Raises
    ------
    DataAbnormalError
        If data contains either multiple measure ids and `single_only` is True
        or a non-permissible measure id.
    """
    if single_only and len(set(data.measure_id)) > 1:
        raise DataAbnormalError(f'Data has multiple measure ids: {set(data.measure_id)}.')
    if not set(data.measure_id).issubset(set([MEASURES[m] for m in allowable_measures])):
        raise DataAbnormalError(f'Data includes a measure id not in the expected measure ids for this measure.')


def check_metric_id(data: pd.DataFrame, expected_metric: str):
    """Check that data contains only a single metric id and that it matches the
    expected metric.

    Parameters
    ----------
    data
        Dataframe containing 'metric_id' column.
    expected_metric
        String dictating the expected metric, the id of which can be found via
        METRICS.

    Raises
    ------
    DataAbnormalError
        If data contains any metric id other than the expected.

    """
    if set(data.metric_id) != {METRICS[expected_metric.capitalize()]}:
        raise DataAbnormalError(f'Data includes metrics beyond the expected {expected_metric.lower()} '
                                f'(metric_id {METRICS[expected_metric.capitalize()]}')


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

