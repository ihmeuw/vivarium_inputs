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

### START OF ALEC'S CODE ###

def makeAppendableData(year_id, sex_id, age_group_id, location_id, fill_na_value):
    index = list(itertools.product(year_id, sex_id, age_group_id,
                                   location_id))  # Makes a list of tuples with all the possible combinations for the demographic columns
    df = pd.DataFrame(
        {'year_id': list(zip(*index))[0], 'sex_id': list(zip(*index))[1], 'age_group_id': list(zip(*index))[2],
         'location_id': list(zip(*index))[3]})  # Makes a dataframe with demographic columns

    draw_columns = [f'draw_{i}' for i in range(0, 1000)]  # Create a list of strings that tells us column names
    df_dummy = pd.DataFrame(data=fill_na_value, index=df.index,
                            columns=draw_columns)  # Create df filled entirely with fill_na_value
    df1 = pd.concat([df, df_dummy], axis=1)  # Concatenate the two data frames (by column)

    return df1


def standardize_data_for_age(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    # Need to extract non-age ids from input data (for later functions)
    this_data_year_ids = set(data['year_id'])
    this_data_sex_ids = set(data['sex_id'])
    this_data_location_ids = set(data['location_id'])

    # Create list of unique age group ids from data and check against extent
    unique_age_group_ids = set(data['age_group_id'])  # Create set of unique age group ids (sets are inherently unique)
    extent = set(extent)  # Make extent a set for subset operations

    #  IF Data has extra age group ids, need to get rid of them
    if not unique_age_group_ids.issubset(extent):
        excess_age_group_ids = unique_age_group_ids.difference(
            extent)  # Isolate excess age group ids from unique_age_group_ids
        for x in excess_age_group_ids:  # Go one-by-one
            unique_age_group_ids.discard(x)
        data = data[data['age_group_id'].isin(unique_age_group_ids)]

    # IF Data needs to be expanded, fill age group ids
    if unique_age_group_ids.issubset(extent) & len(extent) > len(unique_age_group_ids):
        new_age_group_ids = extent.difference(
            unique_age_group_ids)  # Isolate age groups ids that need to be added to data
        appendable_data = makeAppendableData(this_data_year_ids, this_data_sex_ids, new_age_group_ids,
                                             this_data_location_ids, fill_na_value)
        data = pd.concat([data, appendable_data], axis=0)  # Concatenate the two data frames (by row)

    return data

### END OF ALEC'S CODE ###


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


