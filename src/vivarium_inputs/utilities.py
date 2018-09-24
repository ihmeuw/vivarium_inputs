"""Module containing functions that standardize the format of GBD outputs."""
from typing import Mapping

import pandas as pd

from vivarium_inputs.core import UnhandledDataError


def standardize_dimensions(data: pd.DataFrame, dimensions: pd.MultiIndex,
                           fill_na_value: Mapping[str, float]=None) -> pd.DataFrame:
    """Take input data and make it dense over the specified dimensions.

    The behavior of the function depends on which dimension is being considered and the nature of the sparsity:

    Cases:
    1) dimension missing - nothing needed
    2) fully dense in the expected range - nothing needed
    3) some sparsity:
       a) in the year dimension: interpolate
       b) in age/sex dimension and dense for a contiguous subset of the expected range and missing elsewhere: fill
    4) all others - panic

    Parameters
    ----------
    data :
        The data to standardize
    dimensions :
        A multi index whose individual index columns must be some subset of {'age', 'sex', 'year'}
        which represent dimensions of the data; and whose levels provide the expected extent of
        the data in those dimensions.
    fill_na_value :
        A mapping between

    Returns
    -------
        A dataframe with either full extent or no extent in all of the given dimensions.

    Raises
    ------
    UnhandledDataError :
        If the data is somehow malformed in a manner we don't deal with.
    """
    dimensions = dimensions.to_frame().reset_index(drop=True)

    dimensions = pd.concat([dimensions.assign(measure=measure) for measure in data.measure.unique()], ignore_index=True)
    draw_columns = [c for c in data.columns if 'draw_' in c]

    assert set(dimensions.columns) <= {'age_group_id', 'sex', 'year', 'measure', 'location'}
    assert set(data.columns) <= set(dimensions.columns.tolist() + draw_columns)
    assert set(data.measure.unique()) <= fill_na_value.keys()

    # Case 1: Remove any dimensions not present in the data.
    applicable_dimensions = dimensions[[c for c in dimensions.columns if c in data]].copy()
    # Case 4: Check preconditions and panic if it's an unhandled case.
    verify_well_formed(data, applicable_dimensions)

    # Case 3a: Interpolate over year.
    if 'year' in applicable_dimensions.columns:
        data = interpolate_years(data, applicable_dimensions)

    # Case 3b: Fill in missing data
    dimension_columns, extents = zip(*applicable_dimensions.items())
    expected_index = pd.MultiIndex.from_arrays(extents, names=dimension_columns)
    data = data.set_index(list(dimension_columns))

    missing = expected_index.difference(data.index)
    if not missing.empty:
        to_add = pd.DataFrame(columns=draw_columns, index=missing)
        to_add = to_add.reset_index().set_index(list(set(dimension_columns) - {'measure'})).sort_index()
        data = data.reset_index().set_index(list(set(dimension_columns) - {'measure'})).sort_index()
        for measure, fill in fill_na_value.items():
            to_add.loc[to_add['measure'] == measure, draw_columns] = fill
        data = data.append(to_add)

    return data.reset_index()


def verify_well_formed(data: pd.DataFrame, dimensions: pd.DataFrame):
    for dimension in dimensions.columns:
        if dimension == 'sex':
            continue
        existing = data[dimension].sort_values().drop_duplicates()
        expected = dimensions[dimension].sort_values().drop_duplicates()

        contiguous_overlap = ((expected >= existing.min()) & (expected <= existing.max()))
        if dimension == 'age_group_id' and not set(existing) == set(expected[contiguous_overlap]):
            raise UnhandledDataError(f'The data is malformed in the {dimension} dimension.')

        if dimension == 'year' and (existing.min() > expected.min() or existing.max() < expected.max()):
            raise UnhandledDataError("Cannot interpolate years without data on both edges")


def interpolate_years(data: pd.DataFrame, dimensions: pd.DataFrame) -> pd.DataFrame:

    existing_extent = data['year'].sort_values().drop_duplicates()
    expected_extent = dimensions['year'].sort_values().drop_duplicates()

    out = []
    for year in expected_extent:
        if year in existing_extent.values:
            out.append(data[data['year'] == year])
        else:
            from core_maths.interpolate import interpolate
            index_columns = list(dimensions.columns)
            value_columns = list(data.columns.difference(index_columns))

            previous_year = int(existing_extent[existing_extent < year].iloc[-1])
            next_year = int(existing_extent[existing_extent > year].iloc[0])
            start = data[data['year'] == previous_year].sort_values(index_columns).reset_index(drop=True)
            end = data[data['year'] == next_year].sort_values(index_columns).reset_index(drop=True)

            interpolated = interpolate(start, end, index_columns, 'year', value_columns, previous_year, next_year)
            interpolated = interpolated[interpolated['year'] == year]
            out.append(interpolated)

    return pd.concat(out, ignore_index=True)


def select_draw_data(data, draw, column_name, src_column=None):
    if column_name:
        if src_column is not None:
            if isinstance(src_column, str):
                column_map = {src_column.format(draw=draw): column_name}
            else:
                column_map = {src.format(draw=draw): dest for src, dest in zip(src_column, column_name)}
        else:
            column_map = {'draw_{draw}'.format(draw=draw): column_name}

        # if 'measure' is in columns, then keep it, else do
        # not keep it (need measure for the relative risk estimations)
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
    if "sex_id" in df:
        if set(df["sex_id"]) == {3}:
            df_m = df.copy()
            df_f = df.copy()
            df_m['sex'] = 'Male'
            df_f['sex'] = 'Female'
            df = pd.concat([df_m, df_f], ignore_index=True)
        else:
            df["sex"] = df.sex_id.map({1: "Male", 2: "Female", 3: "Both"}).astype(
                pd.api.types.CategoricalDtype(categories=["Male", "Female", "Both"], ordered=False))

        df = df.drop("sex_id", axis=1)
        df = df.rename(columns={"year_id": "year"})
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
    ok for the purposes of Vivarium. Everett proposed that we could get the life expectancy at age 80
    for each location and use that as the midpoint for the 80+ group, but Abie suggested that we
    keep the midpoint as 82.5 for now. GBD populations have data for each age group up until the
    age 95+ age group, at which point I'm assuming we can use 97.5 as the midpoint.
    """
    import vivarium_gbd_access.gbd as gbd
    if df.empty:
        df['age'] = 0
        return df

    df = df.copy()
    idx = df.index
    mapping = gbd.get_age_bins()
    mapping = mapping.set_index('age_group_id')
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)

    df = df.set_index('age_group_id')
    df[['age', 'age_group_start', 'age_group_end']] = mapping[['age', 'age_group_years_start', 'age_group_years_end']]

    # Assumption: We're using 82.5 as the midpoint for the age 80+ age group. May want to change in the future.
    df.loc[df.age == 102.5, 'age'] = 82.5

    df = df.set_index(idx)

    return df
