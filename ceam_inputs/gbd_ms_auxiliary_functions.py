import numpy as np
import pandas as pd

from scipy import stats

from ceam import config

from db_tools import ezfuncs

from ceam_inputs.util import stata_wrapper

# This file contains auxiliary functions that are used
# in gbd_ms_functions.py to prepare data for ceam


# TODO: MAKE SURE NEW PYTHON FUNCTIONS ARE USING THE PUBLICATION IDS!!


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
    # Convert sex_id to a categorical
    df['sex'] = df.sex_id.map(
        {1: 'Male', 2: 'Female', 3: 'Both'}).astype('category')
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
    """

    df = df.copy()
    idx = df.index
    mapping = ezfuncs.query('''select age_group_id, age_group_years_start, age_group_years_end from age_group''', conn_def='shared')
    mapping = mapping.set_index('age_group_id')
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)

    df = df.set_index('age_group_id')
    df['age'] = mapping['age']

    # Assumption: We're using 82.5 as the midpoint for the age 80+ age group. May want to change in the future.
    df.loc[df.age == 102.5, 'age'] = 82.5
    
    # TODO: Confirm this is an ok assumption. GBD produces population data for 80-84, 85-89, and 90-94 year olds. We're seeting the midpoint for the age 95+ year old age group to be 97.5
    df.loc[df.age == 110, 'age'] = 97.5

    df = df.set_index(idx)

    return df


def get_populations(location_id, year_start, sex_id):
    """
    Get age-/sex-specific population structure

    Parameters
    ----------
    location_id : int, location id
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    sex_id: str, sex
        sex_id takes values 1, 2, or 3

    Returns
    -------
    df with columns year_id, location_name, location_id, age, sex_id, and
        pop_scaled
        pop_scaled is the population for a given age/year/sex
    """
    # use central comp's get_population function to get gbd populations
    # the age group id arguments get the age group ids for each age group up through age 95+
    # pop = get_population(age_group_id=list(range(2,21)) + [30, 31, 32] + [235], location_id=location_id, year_id=year_start, sex_id=sex_id)
    pop = stata_wrapper('get_populations.do', 'pop_{l}.csv'.format(l = location_id), location_id)

    # use auxilliary function extract_age_from_age_group_name to create an age
    # column
    pop = get_age_group_midpoint_from_age_group_id(pop)

    # Grab population for year_start only (to initialize microsim population)
    pop = pop.query('year_id=={y}'.format(y=year_start))

    # Determine gender of interest. Can be 1, 2, or 3 (Male, Female, or Both)
    pop = pop.query("sex_id == {s}".format(s=sex_id))

    # Keep only the relevant columns
    pop = pop[['year_id', 'location_name',
               'location_id', 'age', 'sex_id', 'pop_scaled']]

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

    """
    new_sim_file = pd.DataFrame()

    # do for each age in population dataframes (same ages in male_pop and
    # female_pop)
    # FIXME: @alecwd why is the prev_age variable necessary here? I don't think any interpolation has happened at this point.
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


def create_sex_id_column(simulants_df, location_id, year_start):
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
    """

    # Force the year to be a multiple of five because that's the granularity
    # of GBD data
    year_start = int(year_start/5)*5

    # pull in male and female populations so that we can assign sex according
    # to GBD population estimates (with age/sex correlation)
    male_pop = get_populations(location_id, year_start, 1)
    female_pop = get_populations(location_id, year_start, 2)

    # assign sex_id according to proportions calculated from GBD data
    new_sim_file = assign_sex_id(simulants_df, male_pop, female_pop)

    # assert an error to make sure data is dense (i.e. no missing data)
    assert new_sim_file.isnull().values.any() == False, "there are nulls in the dataframe that assign_sex_id just tried to output. check the population file that you pulled in from the GBD database"

    # change sex_id column to be an integer
    new_sim_file.sex_id = new_sim_file.sex_id.astype(int)

    # assert an error that only sex_id 1 and 2 are in the new_sim_file
    assert new_sim_file.sex_id.isin([1, 2]).all() == True, "something went wrong with assign_sex_id. function tried to assign a sex id other than 1 or 2"

    return new_sim_file


def get_all_cause_mortality_rate(location_id, year_start, year_end):
    '''Get cause-deleted mortality rate from year_start to year_end (inclusive)

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    Returns
    -------
    pd.DataFrame with columns
    '''

    # Potential FIXME: Should all_cause_draws and pop be made arguments to the function instead of data grabbed inside the function?
    # TODO: Make this get_draws call more flexible. Currently hardcoded to grab 2015 data.
    all_cause_draws = get_draws(gbd_id_field="cause_id", gbd_id=294, location_ids=location_id, measure_ids=1, source="dalynator", status="best", gbd_round_id=config.getint('simulation_parameters', 'gbd_round_id'))

    # filter so that only metric id 1 (deaths) is in our dataframe
    all_cause_deaths = all_cause_draws.query("metric_id == 1")

    # get population estimates for each age, sex, year group. population estimates will serve
    # as the estimates for # of Person Years in each age group
    pop = get_population(age_group_id=list(range(2,22)), location_id=location_id, year_id=-1, sex_id=[1,2])

    # merge all cause deaths and pop to get all cause mortality rate
    merged = pd.merge(all_cause_deaths, pop, on=[
                            'age_group_id', 'year_id', 'sex_id', 'location_id'])

    # Need to divide # of all cause deaths by population
    # Mortality Rate = Number of Deaths / Person Years of Exposure
    deaths = merged[['draw_{}'.format(i) for i in range(1000)]].values
    population = merged[['pop_scaled']].values

    merged.set_index(['year_id', 'sex', 'age_group_id'], inplace=True)

    all_cause_mr = pd.DataFrame(np.divide(deaths, population), columns=['all_cause_mortality_rate_{}'.format(i) for i in range(0,1000)], index=merged.index)

    all_cause_mortality_rate = get_age_from_age_group_id(all_cause_mr)

    # only get years we care about
    all_cause_mortality_rate = all_cause_mortality_rate.query('year_id>={ys} and year_id<={ye}'.
                                          format(ys=year_start, ye=year_end))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert all_cause_mortality_rate.isnull().values.any() == False, "there are nulls in the dataframe that get_all_cause_mortality just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert all_cause_mortality_rate.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_all_cause_mortality_rate just tried to output. check the cache to make sure that the data you're pulling is correct"

    return output_df


def get_healthstate_id(dis_weight_modelable_entity_id):
    """Returns a healthstate_id for a given modelable entity id

    Parameters
    ----------
    dis_weight_modelable_entity_id : int

    Returns
    -------
    integer specifying healthstate_id for modelable entity id that was supplied
    """
    
    healthstate_id_df = ezfuncs.query('''
    SELECT modelable_entity_id, healthstate_id
    FROM epi.sequela
    WHERE modelable_entity_id = {}
    '''.format(int(dis_weight_modelable_entity_id))
    , conn_def='epi')
    
    if healthstate_id_df.empty:
        raise ValueError("""modelable entity id {} does not have a healthstate id. 
        there is not a disability weight associated with this sequela, 
        so you should not try to pull draws for it""".format(dis_weight_modelable_entity_id))
    
    healthstate_id = healthstate_id_df.at[0,'healthstate_id']

    return healthstate_id


# End.
