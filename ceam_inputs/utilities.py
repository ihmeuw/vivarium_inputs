from typing import Mapping, Iterable


import pandas as pd


from core_maths.interpolate import interpolate

from ceam_inputs.gbd import get_age_bins

def standardize_dimensions(data: pd.DataFrame, dimensions: pd.MultiIndex, fill_na_value: Mapping[str, float]=None) -> pd.DataFrame:
    """
    Take draw data and make it dense over the specified dimensions. The dimensions must be some subset of ['age_group_id', 'sex', 'year'].

    The behavior of the function depends on which dimension is being considered and the nature of the sparcity:

    Cases:
    1) dimension missing - nothing needed
    2) fully dense in the expected range - nothing needed
    3) some spacity:
       in the year dimension: interpolate
       in age/sex dimension and dense for a contiguous subset of the exected range and missing elsewhere: fill
    4) all others - panic
    """

    if fill_na_value is None:
        fill_na_value = {}

    dimensions = dimensions.to_frame().reset_index(drop=True)
    with_parameter = pd.DataFrame()
    for parameter in data.parameter.unique():
        with_parameter = with_parameter.append(dimensions.assign(parameter= parameter))
    dimensions = with_parameter
    assert not set(dimensions.columns).difference(['age_group_id', 'sex', 'year', 'parameter']), "We only know how to standardize 'age_group_id', 'sex' and 'year'"
    draw_columns = {c for c in data.columns if 'draw_' in c}
    assert not set(data.columns).difference(set(dimensions.columns) | {'parameter'} | draw_columns), "We only support standardizing draw data"

    applicable_dimensions = dimensions.copy()

    for dimension in dimensions.columns:
        if dimension not in data:
            # Case 1: completely missing
            del applicable_dimensions[dimension]
            continue

        existing_extent = data[dimension].sort_values().drop_duplicates()
        extent = dimensions[dimension].sort_values().drop_duplicates()

        if existing_extent.equals(extent):
            # Case 2: fully dense
            continue

        if dimension in ('age_group_id', 'sex'):
            min_existing = existing_extent.min()
            max_existing = existing_extent.max()

            overlap = (extent >= min_existing) & (extent <= max_existing)

            if not existing_extent.reset_index(drop=True).equals(extent[overlap].reset_index(drop=True)):
                # Case 4: the data is somehow malformed
                raise ValueError()
        elif dimension == 'year':
            # This one is different because we want to interpolate rather than fill.
            if existing_extent.min() > extent.min() or existing_extent.max() < extent.max():
                raise ValueError("Cannot interpolate years without data on both edges")

            i = 0
            while i < len(extent):
                year = extent.iloc[i]
                if year not in existing_extent.values:
                    j = i
                    while j < len(extent):
                        check_year = extent.iloc[j]
                        if check_year in existing_extent.values:
                            break
                        else:
                            j += 1
                    #TODO does this work if i is the end of the list?
                    year = extent.iloc[i-1]
                    end_year = extent.iloc[j]
                    start = data.query('year == @year').sort_values(list(data.columns)).reset_index(drop=True)
                    end = data.query('year == @end_year').sort_values(list(data.columns)).reset_index(drop=True)

                    index_columns = list(dimensions.columns)
                    value_columns = list(draw_columns)
                    interpolated = interpolate(start, end, index_columns, 'year', value_columns, year, end_year)
                    interpolated = interpolated.query('year != @year and year != @end_year and year in @extent')
                    data = data.append(interpolated)
                i += 1


    # Case 3: fill
    dimension_columns, extents = zip(*applicable_dimensions.items())
    expected_index = pd.MultiIndex.from_arrays(extents, names=dimension_columns)
    data = data.set_index(list(dimension_columns))

    missing = expected_index.difference(data.index)
    if not missing.empty:
        to_add = pd.DataFrame(columns=list(draw_columns), index=missing)
        to_add = to_add.reset_index().set_index(list(set(dimension_columns) - {'parameter'}))
        data = data.reset_index().set_index(list(set(dimension_columns) - {'parameter'}))
        for parameter, fill in fill_na_value.items():
            to_add.loc[to_add['parameter'] == parameter, draw_columns] = fill
        data = data.append(to_add)

    return data.reset_index()


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
