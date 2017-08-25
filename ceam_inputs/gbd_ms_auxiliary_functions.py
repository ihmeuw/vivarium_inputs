import numpy as np
import pandas as pd

from ceam_inputs import gbd


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
    df['sex'] = df.sex_id.map({1: 'Male', 2: 'Female', 3: 'Both'}).astype('category',
                                                                          categories=['Male', 'Female', 'Both'])
    df = df.drop('sex_id', axis=1)
    df = df.rename(columns={'year_id': 'year'})
    return df


def get_age_group_midpoint_from_age_group_id(df, gbd_round_id):
    """Creates an "age" column from the "age_group_id" column

    Parameters
    ----------
    df: df for which you want an age column that has an age_group_id column
    gbd_round_id: int

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
    mapping = gbd.get_age_bins(gbd_round_id)
    mapping = mapping.set_index('age_group_id')
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)

    df = df.set_index('age_group_id')
    df[['age', 'age_group_start', 'age_group_end']] = mapping[['age', 'age_group_years_start', 'age_group_years_end']]

    # Assumption: We're using 82.5 as the midpoint for the age 80+ age group. May want to change in the future.
    df.loc[df.age == 102.5, 'age'] = 82.5

    df = df.set_index(idx)

    return df


def expand_grid(a, y):
    """
    Creates an expanded dataframe of ages and years
    Mirrors the expand_grid function in R
    See http://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python
    for more details

    Parameters
    ----------
    a: age values that you on which you want to expand
    y: year values that you on which you want to expand

    Returns
    -------
    Dataframe of expanded ages and years
    """

    ages, years = np.meshgrid(a, y)  # create the actual grid
    ages = ages.flatten()  # make the grid 1d
    years = years.flatten()  # make the grid 1d
    df = pd.DataFrame({'age': ages, 'year_id': years})  # return a dataframe
    return df[['year_id', 'age']].sort_values(by=['year_id', 'age'])


def get_all_age_groups_for_years_in_df(df, gbd_round_id):
    """Returns a dataframe with all ages for all years in df

    columns are age and year_id
    """

    mapping = gbd.get_age_bins(gbd_round_id)
    mapping_filter = mapping.query('age_group_id >=2 and age_group_id <=21').copy()
    mapping_filter['age'] = mapping_filter[['age_group_years_start', 'age_group_years_end']].mean(axis=1)
    mapping_filter.loc[mapping_filter.age == 102.5, 'age'] = 82.5

    expanded_data = expand_grid(mapping_filter.age.values, pd.unique(df.year_id.values))

    return expanded_data


def expand_ages(df, gbd_round_id):
    """
    Parameters
    ----------
    df: pandas.DataFrame
    gbd_round_id: int

    Returns
    -------
    pandas.DataFrame
    """
    expanded_cols = get_all_age_groups_for_years_in_df(df, gbd_round_id)
    expanded_indexed = expanded_cols.set_index(['year_id', 'age'])
    indexed_df = df.set_index(['year_id', 'age']).sort_index(level=0)
    total_df = pd.DataFrame()

    # for rrs and exposures, there are multiple parameters
    if 'parameter' in df.columns:
        for sex in pd.unique(df.sex_id.values):
            for param in pd.unique(df.parameter.values):
                one_df = pd.DataFrame(indexed_df.query('parameter == @param and sex_id == @sex'),
                                      index=expanded_indexed.index)
                one_df['sex_id'] = sex
                one_df['parameter'] = param
                total_df = total_df.append(one_df)

    else:
        for sex in pd.unique(df.sex_id.values):
            one_df = pd.DataFrame(indexed_df.query('sex_id == @sex'), index=expanded_indexed.index)
            one_df['sex_id'] = sex
            total_df = total_df.append(one_df)

    return total_df.reset_index()


def expand_ages_for_dfs_w_all_age_estimates(df, gbd_round_id):
    """Some tables only have estimates for the "all ages" age group instead of
    broken out for the different age group ids. We need estimates for each age
    group separately, so this function expands tables with estimates for only
    the "all ages" to include estimates for each age group id
    """
    expanded_cols = get_all_age_groups_for_years_in_df(df, gbd_round_id)
    male_data = pd.merge(df, expanded_cols)
    male_data['sex_id'] = 1
    female_data = pd.merge(df, expanded_cols)
    female_data['sex_id'] = 2
    total_data = male_data.append(female_data)
    return total_data
