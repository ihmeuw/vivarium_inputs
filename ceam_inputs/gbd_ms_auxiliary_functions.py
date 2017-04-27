import numpy as np
import pandas as pd

from scipy import stats

from ceam import config

try:
    from db_tools.ezfuncs import query
except ImportError:
    def query(*args, **kwarg):
        raise ImportError("No module named 'db_tools' (you must install central_comp's db_tools package _or_ supply precached data)")

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
    Returns a df with column year_id changed to year, sex_id changed to sex, and sex values changed from 1 and 2 to Male and Female

    Notes
    -----
    Used by -- load_data_from_cache

    Assumptions -- None

    Questions -- None

    Unit test in place? -- Yes
    """

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

    Notes
    -----
    Used by -- get_modelable_entity_draws, get_relative_risks, get_pafs, get_exposures, get_sbp_mean_sd, get_bmi_distributions, get_age_specific_fertility_rates, get_populations

    Assumptions -- We assume that using a midpoint of age 82.5 for the 80+ year old age group is ok for the purposes of CEAM. Everett proposed that we could get the life expectancy at age 80 for each location and use that as the midpoint for the 80+ group, but Abie suggested that we keep the midpoint as 82.5 for now. GBD populations have data for each age group up until the age 95+ age group, at which point I'm assuming we can use 97.5 as the midpoint.

    Questions -- None

    Unit test in place? -- Yes
    """
    if df.empty:
        df['age'] = 0
        return df

    df = df.copy()
    idx = df.index
    mapping = query('''select age_group_id, age_group_years_start, age_group_years_end from age_group''', conn_def='shared')
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

# TODO: Make the central comp get_populations call take a gbd_round_id argument
def get_populations(location_id, year_start, sex_id, get_all_years=False, sum_up_80_plus=False):
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

    Notes
    -----
    Used by -- generate_ceam_population, create_sex_id_column

    Assumptions --  None

    Questions -- None

    Unit test in place? -- No. Don't think one is needed. We just use the central comp get_population function to get the population data and then select a specific year, specific sex, and use the get_age_group_midpoint_from_age_group_id function to get the age group midpoints.

    Uncertainty draws -- Need to be cognizant of the fact that there are not currently uncertainty estimates for populations in GBD, but that these estimates will be produced for GBD 2017, and maybe even GBD 2016. Hopefully, when the draws are ready, we will be able to continue using central comp's get_populations function.
    """
    from db_queries import get_population # This import is not at global scope because I only want the dependency if cached data is unavailable

    # use central comp's get_population function to get gbd populations
    # the age group id arguments get the age group ids for each age group up through age 95+
    if get_all_years:
        year_id = -1
    else:
        # Grab population for year_start only (to initialize microsim population)
        year_id = year_start
    pop = get_population(age_group_id=list(range(2,21)) + [30, 31, 32] + [235], location_id=location_id, year_id=year_id, sex_id=sex_id)

    # Don't include the older ages in the 2015 runs
    # if config.simulation_parameters.gbd_round_id == 3:
    #    pop = pop.query("age_group_id <= 21")

    # use auxilliary function extract_age_from_age_group_name to create an age
    # column
    pop = get_age_group_midpoint_from_age_group_id(pop)

    # Determine gender of interest. Can be 1, 2, or 3 (Male, Female, or Both)
    pop = pop.query("sex_id == {s}".format(s=sex_id))

    # Keep only the relevant columns
    pop = pop[['year_id', 
               'location_id', 'age', 'sex_id', 'population']]

    # The population column was called pop_scaled in GBD 2015, but name was changed. Changing it back since all of our code uses pop_scaled as the col name
    pop = pop.rename(columns={'population': 'pop_scaled'})

    # FIXME: As of 1-23, get_populations is only function we use that has data for detailed 5 year age groups over age of 80. We need to get the 80+ age group to make data compatible with other data, but will likely not need this in the future if all other estimates start giving data for more detailed age groups over the age of 80
    # if config.simulation_parameters.gbd_round_id != 3:
    if sum_up_80_plus == True:
        older_pop = pop.query("age >= 80").copy()

        older_grouped = older_pop.groupby(['year_id', 'location_id', 'sex_id'], as_index=False).sum()

        older_grouped['age'] = 82.5

        younger_pop = pop.query("age < 80").copy()

        del(pop)

        pop = younger_pop.append(older_grouped)        

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


def get_all_cause_mortality_rate(location_id, year_start, year_end, gbd_round_id):
    '''Get cause-deleted mortality rate from year_start to year_end (inclusive)

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    gbd_round_id: int
        GBD round to pull data for

    Returns
    -------
    pd.DataFrame with columns

    Notes
    -----
    Used by -- get_cause_deleted_mortality_rate

    Assumptions -- None

    Questions -- Is the dalynator the correct source for pulling the all-cause mortality rate? 

    Unit test in place? -- Not currently, but one does need to be put in place
    '''
    from transmogrifier.draw_ops import get_draws # This import is not at global scope because I only want the dependency if cached data is unavailable

    # Potential FIXME: Should all_cause_draws and pop be made arguments to the function instead of data grabbed inside the function?
    # TODO: Make this get_draws call more flexible. Currently hardcoded to grab 2015 data.
    worker_count = int((year_end - year_start)/5) # One worker per 5-year dalynator file
    all_cause_draws = get_draws(gbd_id_field="cause_id", gbd_id=294, age_group_ids=list(range(2,22)), location_ids=location_id, measure_ids=1, source="dalynator", status="best", gbd_round_id=gbd_round_id, year_ids=range(year_start, year_end+1), num_workers=worker_count)

    # filter so that only metric id 1 (deaths) is in our dataframe
    all_cause_deaths = all_cause_draws.query("metric_id == 1")

    all_cause_deaths = all_cause_deaths.query("sex_id != 3")

    all_cause_deaths = get_age_group_midpoint_from_age_group_id(all_cause_deaths)

    # get population estimates for each age, sex, year group. population estimates will serve
    # as the estimates for # of Person Years in each age group
    # pop = get_populations(age_group_id=list(range(2,22)), location_id=location_id, year_id=-1, sex_id=[1,2])
    pop_male = get_populations(location_id, year_start, 1, get_all_years=True, sum_up_80_plus=True)
    pop_female = get_populations(location_id, year_start, 2, get_all_years=True, sum_up_80_plus=True)

    pop = pop_male.append(pop_female)

    # merge all cause deaths and pop to get all cause mortality rate
    merged = pd.merge(all_cause_deaths, pop, on=[
                            'age', 'year_id', 'sex_id', 'location_id'])

    # Need to divide # of all cause deaths by population
    # Mortality Rate = Number of Deaths / Person Years of Exposure
    deaths = merged[['draw_{}'.format(i) for i in range(1000)]].values
    population = merged[['pop_scaled']].values

    merged.set_index(['year_id', 'sex_id', 'age'], inplace=True)

    all_cause_mr = pd.DataFrame(np.divide(deaths, population), columns=['all_cause_mortality_rate_{}'.format(i) for i in range(0,1000)], index=merged.index)

    all_cause_mr = all_cause_mr.reset_index()

    # only get years we care about
    all_cause_mortality_rate = all_cause_mr.query('year_id>=@year_start and year_id<=@year_end')

    # assert an error to make sure data is dense (i.e. no missing data)
    assert all_cause_mortality_rate.isnull().values.any() == False, "there are nulls in the dataframe that get_all_cause_mortality just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert all_cause_mortality_rate.duplicated(['age', 'year_id', 'sex_id']).sum() == 0, "there are duplicates in the dataframe that get_all_cause_mortality_rate just tried to output. check the cache to make sure that the data you're pulling is correct"

    return all_cause_mortality_rate


def get_healthstate_id(dis_weight_modelable_entity_id):
    """Returns a healthstate_id for a given modelable entity id

    Parameters
    ----------
    dis_weight_modelable_entity_id : int

    Returns
    -------
    integer specifying healthstate_id for modelable entity id that was supplied

    Used by -- get_disability_weight

    Assumptions -- None

    Questions -- None

    Unit test in place? -- Yes
    """
    
    healthstate_id_df = query('''
    SELECT modelable_entity_id, healthstate_id
    FROM epi.sequela_hierarchy_history
    WHERE modelable_entity_id = {}
    '''.format(int(dis_weight_modelable_entity_id))
    , conn_def='epi')
    
    if healthstate_id_df.empty:
        raise ValueError("""modelable entity id {} does not have a healthstate id. 
        there is not a disability weight associated with this sequela, 
        so you should not try to pull draws for it""".format(dis_weight_modelable_entity_id))
    
    healthstate_id = healthstate_id_df.at[0,'healthstate_id']

    return healthstate_id


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

    aG, yG = np.meshgrid(a, y)  # create the actual grid
    aG = aG.flatten()  # make the grid 1d
    yG = yG.flatten()  # make the grid 1d
    df = pd.DataFrame({'age': aG, 'year_id': yG})  # return a dataframe
    return df[['year_id', 'age']].sort_values(by=['year_id', 'age'])


def get_all_age_groups_for_years_in_df(df):
    """
    returns a dataframe with all ages for all years in df

    columns are age and year_id
    """
    mapping = query('''select age_group_id, age_group_years_start, age_group_years_end from age_group''', conn_def='shared')
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
                one_df = pd.DataFrame(indexed_df.query('parameter == @param and sex_id == @sex'), index=expanded_indexed.index)
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
    '''
    Some tables only have estimates for the "all ages" age group instead of broken out for the different age group ids. We need estimates for each age group separately, so this function expands tables with estimates for only the "all ages" to include estimates for each age group id
    '''
    expanded_cols = get_all_age_groups_for_years_in_df(df)

    male_data = pd.merge(df, expanded_cols)

    male_data['sex_id'] = 1

    female_data = pd.merge(df, expanded_cols)

    female_data['sex_id'] = 2

    total_data = male_data.append(female_data)

    return total_data


# End.
