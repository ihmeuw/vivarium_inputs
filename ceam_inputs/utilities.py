from typing import Mapping, Iterable
from scipy import interpolate
import pandas as pd

from ceam_inputs.gbd import get_age_bins


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

### START OF PAOLA'S CODE ###

def makeMissingData(data_frame, extent, fill_na_value):
    data = data_frame.copy()
    extent_missing_data = extent  # assume this is an input to the standardize_data_for_age function
    age_ids_in_data = data['age_group_id'].unique()
    missing_age_group_ids = extent_missing_data.difference(age_ids_in_data)
    year_ids = set(data['year_id'])
    location_ids = set(data['location_id'])
    sex_ids = set(data['sex_id'])
    age_group_ids = missing_age_group_ids
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    value = fill_na_value
    index = list(itertools.product(year_ids, sex_ids, age_group_ids,
                                   location_ids))  # makes a list of tuples with all the possible combinations for the demographic columns
    df = pd.DataFrame(
        {'year_id': list(zip(*index))[0], 'sex_id': list(zip(*index))[1], 'age_group_id': list(zip(*index))[2],
         'location_id': list(zip(*index))[3]})
    df_dummy = pd.DataFrame(data=value, index=df.index, columns=draw_columns)
    df1 = pd.concat([df, df_dummy], axis=1)
    return df1


def check_for_age_group_ids(data_frame, extent, value):
    df_ids = set(data['age_group_id'])  # generates a set with age_group_ids in the df
    extent = set(extent)  # turns list into set

    # in this case data has too many age_group_ids
    if not df_ids.issubset(extent):
        # make a list of excess elements
        extra_age_group_ids = df_ids.difference(extent)
        # remove additional age_group_ids from data.
        for x in extra_age_group_ids:
            df_ids.discard(x)
    if df_ids.issubset(extent):
        clean_data = data[data['age_group_id'].isin(df_ids)]

    if len(extent) > len(df_ids):
        missing_data = makeMissingData(clean_data, extent, value)
        data_with_correct_age_group_ids = pd.concat([missing_data, clean_data], axis=0)

    return data_with_correct_age_group_ids

### END OF PAOLA'S CODE ###


### ALEC'S CODE ###
### Make dummy data again, but this time with specific values that we need to add to dummy data
def makeAppendableData(year_id, sex_id, age_group_id, location_id, fill_na_value):
    index = list(itertools.product(year_id, sex_id, age_group_id,
                                   location_id))  # Makes a list of tuples with all the possible combinations for the
    #  demographic columns
    df = pd.DataFrame(
        {'year_id': list(zip(*index))[0], 'sex_id': list(zip(*index))[1], 'age_group_id': list(zip(*index))[2],
         'location_id': list(zip(*index))[3]})  # Makes a dataframe with demographic columns

    draw_columns = [f'draw_{i}' for i in range(0, 1000)]  # Create a list of strings that tells us column names
    df_dummy = pd.DataFrame(data=fill_na_value, index=df.index,
                            columns=draw_columns)  # Create df filled entirely with fill_na_value
    df1 = pd.concat([df, df_dummy], axis=1)  # Concatenate the two data frames (by column)

    return df1


def standardize_data_for_sex(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    # Potential inputs:
    # Data with all sex_ids -> return data
    # Data with only sex_id 1 or sex_id 2 -> Generate other half and sex_id 3, fill_na
    # Data with only sex_id 1 and sex_id 2 -> Population weighted average for sex_id 3 (Unlikely to occur) (HOLD)
    # Data with only sex_id 3 -> Drop sex_id column.

    possible_sex_ids = set({1, 2, 3})  # All possible sex ids
    sex_ids_in_data = set(data['sex_id'])  # The sex ids that are present in the data
    missing_sex_ids = possible_sex_ids.difference(sex_ids_in_data)  # Missing sex ids
    if len(missing_sex_ids) == 0:  # Nothing is missing, just pass data
        data = data

    if len(missing_sex_ids) == 1:  # One sex_id is missing
        if 3 not in sex_ids_in_data:
            # Weighted average for sex_id 3 (HOLD)
            print('on hold')

        else:
            # Generate [missing_sex_id] and fill na
            # Just use the lone value that is in missing_sex_ids as input to generator
            appendable_data = makeAppendableData(set(data['year_id']), missing_sex_ids, set(data['age_group_id']),
                                                 set(data['location_id']), fill_na_value)
            data = pd.concat([data, appendable_data], axis=0)

    if len(missing_sex_ids) == 2:  # Two sex_ids are missing
        if 3 in sex_ids_in_data:
            # Drop sex_id column
            del data['sex_id']
        else:
            # Generate [missing_sex_id_1] and [missing_sex_id_2] and fill na
            # Just use the two values that are in missing_sex_ids as input to generator
            appendable_data = makeAppendableData(set(data['year_id']), missing_sex_ids, set(data['age_group_id']),
                                                 set(data['location_id']), fill_na_value)
            data = pd.concat([data, appendable_data], axis=0)

    ## Check against extent and get rid of excess
    # (James said we may not need this but it's already written so leaving it for now)
    sex_ids_to_cut = set(data['sex_id']).difference(extent)  # Find which sex ids need to be removed
    sex_ids_to_keep = possible_sex_ids  # Start with keeping all sex ids
    for i in sex_ids_to_cut:
        sex_ids_to_keep.discard(i)
    data = data[data['sex_id'].isin(sex_ids_to_keep)]

    ## Check that sex ids are 1, 2, or 3
    assert set(data['sex_id']).issubset({1, 2, 3})

    return data
### END ALEC'S CODE ###

### START OF PAOLA'S CODE (SEX_ID) ###
def standardize_data_for_sex(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
    standard_sex_group_ids = [1, 2, 3]
    sex_ids_in_df = data['sex_id'].unique()

    males = 1
    females = 2
    both = 3

    if len(sex_ids_in_df) == 3:
        data = data.copy()  # return original dataframe

    if len(sex_ids_in_df) == 2:
        print('FIXME: df only has values for males and females. Missing Both')

    if len(sex_ids_in_df) == 1:
        # make dataframe with missing values and append it to the original.

        if sex_ids_in_df == both:
            data = data.drop('sex_id', 1)

        else:
            missing_sex_ids = set(standard_sex_group_ids).difference(sex_ids_in_df)
            year_ids = set(data['year_id'])
            location_ids = set(data['location_id'])
            age_group_ids = set(data['age_group_id'])
            draw_columns = [f'draw_{i}' for i in range(0, 1000)]
            index = list(itertools.product(year_ids, missing_sex_ids, age_group_ids,
                                           location_ids))  # makes a list of tuples with all the possible combinations for the demographic columns
            df = pd.DataFrame(
                {'year_id': list(zip(*index))[0], 'sex_id': list(zip(*index))[1], 'age_group_id': list(zip(*index))[2],
                 'location_id': list(zip(*index))[3]})
            df_dummy = pd.DataFrame(data=fill_na_value, index=df.index, columns=draw_columns)
            df1 = pd.concat([df, df_dummy], axis=1)
            data = pd.concat([data, df1], axis=0)

    # checks the sex_ids of the resulting dataframe
    if 'sex_id' in data:
        assert (set(data['sex_id']).issubset(set(standard_sex_group_ids))), "sex_ids!"

    return data  # or data[data['year_id'].isin(extent)]
### END OF PAOLA'S CODE ###

### START OF PAOLA'S CODE (YEAR_ID) ###

def standardize_data_for_year(data: pd.DataFrame, fill_na_value: float, extent: Iterable[int]) -> pd.DataFrame:
        standard_years = set(range(1990, 2017))
        years_in_df = set(data['year_id'])

        if years_in_df.issuperset(standard_years):
            data = data[data['year_id'].isin(standard_years)]

        else:

            year_ids = standard_years.difference(years_in_df)
            sex_ids = set(data['sex_id'])
            age_group_ids = set(data['age_group_id'])
            location_ids = set(data['location_id'])
            index = list(itertools.product(year_id, sex_id, age_group_id,
                                           location_id))  # makes a list of tuples with all the possible combinations for the demographic columns
            empty_df = pd.DataFrame(
                {'year_id': list(zip(*index))[0], 'sex_id': list(zip(*index))[1], 'age_group_id': list(zip(*index))[2],
                 'location_id': list(zip(*index))[3]})  # makes a dataframe with demographic columns
            df_all_years = pd.concat([data, empty_df], axis=0)
            data = df_all_years.interpolate(method='linear')

        return data  # or data[data['year_id'].isin(extent)]

### END OF PAOLA'S CODE ###

def select_draw_data(data, draw, column_name, src_column=None):
    if column_name:
        if src_column is not None:
            if isinstance(src_column, str):
                column_map = {src_column.format(draw=draw): column_name}
            else:
                column_map = {src.format(draw=draw): dest for src, dest in zip(src_column, column_name)}
        else:
            column_map = {'draw_{draw}'.format(draw=draw): column_name}

        # if 'parameter' is in columns, then keep it, else do
        # not keep it (need parameter for the relative risk estimations)
        if 'parameter' in data.columns:
            keep_columns = ['year_id', 'age', 'sex_id', 'parameter'] + list(column_map.keys())
        else:
            keep_columns = ['year_id', 'age', 'sex_id'] + list(column_map.keys())

        data = data[keep_columns]
        data = data.rename(columns=column_map)

        return normalize_for_simulation(data)
    return data

def normalize_for_simulation(df):
    """
    Parameters
    ----------
    df : DataFrame
        dataframe to change

    Returns
    -------
    Returns a df with column year_id changed to year,
    sex_id changed to sex, and sex values changed from 1 and 2 to Male and Female

    Notes
    -----
    Used by -- load_data_from_cache

    Assumptions -- None

    Questions -- None

    Unit test in place? -- Yes
    """
    df['sex'] = df.sex_id.map({1: 'Male', 2: 'Female', 3: 'Both'}).astype(
        pd.api.types.CategoricalDtype(categories=['Male', 'Female', 'Both'], ordered=False))
    df = df.drop('sex_id', axis=1)
    df = df.rename(columns={'year_id': 'year'})
    return df


def get_age_group_midpoint_from_age_group_id(df):
    """Creates an "age" column from the "age_group_id" column

    Parameters
    ----------
    df: df for which you want an age column that has an age_group_id column

    Returns
    -------
    df with an age column

    Notes
    -----
    Assumptions -- We assume that using a midpoint of age 82.5 for the 80+ year old age group is
    ok for the purposes of CEAM. Everett proposed that we could get the life expectancy at age 80
    for each location and use that as the midpoint for the 80+ group, but Abie suggested that we
    keep the midpoint as 82.5 for now. GBD populations have data for each age group up until the
    age 95+ age group, at which point I'm assuming we can use 97.5 as the midpoint.
    """
    if df.empty:
        df['age'] = 0
        return df

    df = df.copy()
    idx = df.index
    mapping = get_age_bins()
    mapping = mapping.set_index('age_group_id')
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)

    df = df.set_index('age_group_id')
    df[['age', 'age_group_start', 'age_group_end']] = mapping[['age', 'age_group_years_start', 'age_group_years_end']]

    # Assumption: We're using 82.5 as the midpoint for the age 80+ age group. May want to change in the future.
    df.loc[df.age == 102.5, 'age'] = 82.5

    df = df.set_index(idx)

    return df
