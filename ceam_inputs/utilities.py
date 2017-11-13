from typing import Mapping, Iterable

import pandas as pd


def standardize_data_shape(data: pd.DataFrame, fill_na_value: float,
                           extent_mapping: Mapping[str, Iterable[int]]) -> pd.DataFrame:
    function_map = {'age_group_id': standardize_data_for_age,
                    'sex_id': standardize_data_for_sex,
                    'year_id': standardize_data_for_year, }
    for key_column in extent_mapping:
        data = function_map[key_column](data, fill_na_value, extent_mapping[key_column])
    return data


def standardize_data_for_age(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    # Potential inputs:
    # Data with all age groups and all data.
    # Data with some age groups that needs to be filled with an appropriate null value.
    # (Unlikely) Data with scattered age groups that needs interpolation. Don't handle this case
    # Data with a single age group representing all ages (probably -1 in the age group id column) that needs the age
    # group id column removed.
    #
    return data


def standardize_data_for_sex(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    # Potential inputs:
    # Data with all sex_ids -> return data
    # Data with only sex_id 1 or sex_id 2 -> Generate other half and sex_id 3, fill_na
    # Data with only sex_id 1 and sex_id 2 -> Population weighted average for sex_id 3 (Unlikely to occur)
    # Data with only sex_id 3 -> Drop sex_id column.
    return data


def standardize_data_for_year(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    # Potential inputs:
    # Data with all year ids and all data -> return data
    # Data with too many year_ids -> clip and return
    # Data with only gbd_years -> interpolate(1d)
    return data


