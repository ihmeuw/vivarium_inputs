import numpy as np
import pandas as pd

from ceam import config
from ceam_inputs import gbd


def create_age_column(simulants_file, population_file, number_of_simulants):
    """
    Returns a df with a simulant_id and age column

    Parameters
    ----------
    simulants_file : df
        dataframe onto which we want to add an age column

    population_file : df
        population file for location/year of interest for both sexes

    number_of_simulants : int
        number of simulants in simulants_file

    Returns
    -------
    df with columns simulant_id and age

    Notes
    -----
    Used by -- generate_ceam_population

    Assumptions -- None

    Questions - None

    Unit test in place? -- Yes
    """

    # use stats package to assign ages to simulants according to proportions in
    # the population file
    # TODO: potential improvement could be to use np.random.choice and assign
    # age/sex at the same time

    ages = population_file.age.values
    proportions = population_file.proportion_of_total_pop.values
    simulant_ages = np.random.choice(ages, number_of_simulants, p=list(proportions))
    simulants_file['age'] = simulant_ages

    return simulants_file


def normalize_for_simulation(df):
    """
    Parameters
    ----------
    df : pd.DataFrame
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
    df['sex'] = df.sex_id.map({1: 'Male', 2: 'Female', 3: 'Both'}).astype('category')
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
    mapping = gbd.get_age_bins()
    mapping = mapping.set_index('age_group_id')
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)

    df = df.set_index('age_group_id')
    df['age'] = mapping['age']

    # Assumption: We're using 82.5 as the midpoint for the age 80+ age group. May want to change in the future.
    df.loc[df.age == 102.5, 'age'] = 82.5

    # TODO: Confirm this is an ok assumption. GBD produces population data for 80-84, 85-89, and 90-94 year olds.
    # We're setting the midpoint for the age 95+ year old age group to be 97.5
    df.loc[df.age == 110, 'age'] = 97.5

    df = df.set_index(idx)

    return df


def get_populations(location_id, year_start, sex_id, gbd_round_id, get_all_years=False):
    """Get age-/sex-specific population structure

    Parameters
    ----------
    location_id : int, location id
        location_id takes same location_id values as are used for GBD
    year_start : int, year
        year_start is the year in which you want to start the simulation
    sex_id: str, sex
        sex_id takes values 1, 2, or 3
    get_all_years: bool

    Returns
    -------
    df with columns year_id, location_name, location_id, age, sex_id, and pop_scaled
        pop_scaled is the population for a given age/year/sex

    Notes
    -----
    Unit test in place? -- No. Don't think one is needed. We just use the central comp get_population
    function to get the population data and then select a specific year, specific sex, and use the
    get_age_group_midpoint_from_age_group_id function to get the age group midpoints.
    Uncertainty draws -- Need to be cognizant of the fact that there are not currently uncertainty
    estimates for populations in GBD, but that these estimates will be produced for GBD 2017, and
    maybe even GBD 2016. Hopefully, when the draws are ready, we will be able to continue using
    central comp's get_populations function.
    """
    pop = gbd.get_populations(location_id, gbd_round_id=gbd_round_id)
    pop = pop[pop.sex_id == sex_id]

    if not get_all_years:
        pop = pop[pop.year_id == year_start]

    pop = get_age_group_midpoint_from_age_group_id(pop)
    pop = pop.query("sex_id == {s}".format(s=sex_id))
    pop = pop[['year_id', 'location_id', 'age', 'sex_id', 'population']]

    # The population column was called pop_scaled in GBD 2015, but name was changed.
    # Changing it back since all of our code uses pop_scaled as the col name
    pop = pop.rename(columns={'population': 'pop_scaled'})

    # assert an error if there are duplicate rows
    assert pop.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_populations just tried to output"

    # assert an error to make sure data is dense (i.e. no missing data)
    assert pop.isnull().values.any() == False, "there are nulls in the dataframe that get_populations just tried to output"

    return pop


def assign_sex_id(simulants_df, male_pop, female_pop):
    """
    Assigns sex to a population of simulants so that age and sex are correlated

    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made earlier in the function

    male_pop : dataframe
        dataframe containing info on a male pop in location of interest

    female_pop : dataframe
        dataframe containing info on a female pop in location of interest

    Returns
    -------
    A dataframe with a sex_id column with age/sex correlated

    Notes
    -----
    Used by -- create_sex_id_column

    Assumptions -- That we can assign ages/sexes at different times while still ensuring correlation.

    Questions -- Currently, when we generate a population of simulants, we assign age and then assign sex. Should we be assigning age and sex at the same time?
 
    Unit test in place? -- Yes
    """

    new_sim_file = pd.DataFrame()

    # do for each age in population dataframes (same ages in male_pop and
    # female_pop)
    prev_age = -1
    for age in male_pop.age.values:
        male_pop_value = male_pop.query(
            "age > @prev_age and age <= @age").pop_scaled.values
        female_pop_value = female_pop.query(
            "age > @prev_age and age <= @age").pop_scaled.values

        elements = [1, 2]
        male_prop = male_pop_value / (male_pop_value + female_pop_value)
        female_prop = 1 - male_prop
        weights = [float(male_prop), float(female_prop)]

        one_age = simulants_df.query('age > @prev_age and age <= @age').copy()
        one_age['sex_id'] = one_age['age'].map(
            lambda x: np.random.choice(elements, p=weights))

        new_sim_file = new_sim_file.append(one_age)

        prev_age = age

    return new_sim_file


def create_sex_id_column(simulants_df, location_id, year_start, gbd_round_id):
    """
    creates a sex_id column and ensures correlation between age and sex

    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made earlier in the function

    location_id : int, location id
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    Returns
    -------
    Produces a dataframe with sex values
        Sex values are correlated with age

    Used by-- generate_ceam_population

    Assumptions -- That we can assign ages/sexes at different times while still ensuring correlation.

    Questions -- Currently, when we generate a population of simulants, we assign age and then assign sex. Should we be assigning age and sex at the same time?

    Unit test in place? -- No. Don't think it's needed for this function, since this function just utilizes two of our other functions (get_populations and assign_sex_id) which are already tested.
    """

    # Force the year to be a multiple of five because that's the granularity
    # of GBD data
    year_start = int(year_start/5)*5

    # pull in male and female populations so that we can assign sex according
    # to GBD population estimates (with age/sex correlation)
    male_pop = get_populations(location_id, year_start, 1, gbd_round_id)
    female_pop = get_populations(location_id, year_start, 2, gbd_round_id)

    # assign sex_id according to proportions calculated from GBD data
    new_sim_file = assign_sex_id(simulants_df, male_pop, female_pop)

    # assert an error to make sure data is dense (i.e. no missing data)
    assert new_sim_file.isnull().values.any() == False, "there are nulls in the dataframe that assign_sex_id just tried to output. check the population file that you pulled in from the GBD database"

    # change sex_id column to be an integer
    new_sim_file.sex_id = new_sim_file.sex_id.astype(int)

    # assert an error that only sex_id 1 and 2 are in the new_sim_file
    assert new_sim_file.sex_id.isin([1, 2]).all() == True, "something went wrong with assign_sex_id. function tried to assign a sex id other than 1 or 2"

    return new_sim_file


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


def get_all_age_groups_for_years_in_df(df):
    """Returns a dataframe with all ages for all years in df

    columns are age and year_id
    """
    mapping = gbd.get_age_bins()
    mapping_filter = mapping.query('age_group_id >=2 and age_group_id <=21').copy()
    mapping_filter['age'] = mapping_filter[['age_group_years_start', 'age_group_years_end']].mean(axis=1)
    mapping_filter.loc[mapping_filter.age == 102.5, 'age'] = 82.5

    expanded_data = expand_grid(mapping_filter.age.values, pd.unique(df.year_id.values))

    return expanded_data


def expand_ages(df):
    expanded_cols = get_all_age_groups_for_years_in_df(df)
    expanded_indexed = expanded_cols.set_index(['year_id', 'age'])
    indexed_df = df.set_index(['year_id', 'age']).sortlevel()
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


def expand_ages_for_dfs_w_all_age_estimates(df):
    """Some tables only have estimates for the "all ages" age group instead of 
    broken out for the different age group ids. We need estimates for each age 
    group separately, so this function expands tables with estimates for only 
    the "all ages" to include estimates for each age group id
    """
    expanded_cols = get_all_age_groups_for_years_in_df(df)
    male_data = pd.merge(df, expanded_cols)
    male_data['sex_id'] = 1
    female_data = pd.merge(df, expanded_cols)
    female_data['sex_id'] = 2
    total_data = male_data.append(female_data)
    return total_data
