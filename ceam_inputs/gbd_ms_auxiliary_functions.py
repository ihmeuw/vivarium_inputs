import numpy as np
import pandas as pd

from scipy import stats

from ceam import config

from db_tools import ezfuncs

from ceam_inputs.util import stata_wrapper

# This file contains auxiliary functions that are used
# in gbd_ms_functions.py to prepare data for ceam


def set_age_year_index(df, age_start, age_end, year_start, year_end):
    """
    Return a dataframe with age and year indexes. Preps the data for
        interpolation

    Parameters
    ----------
    df: df
        df contains raw inputs from GBD, so the data only contains info
        for age groups and GBD years

    age_start: int or str
        earliest age for which you want data
            NOTE: If you want your age range to start with early, late, or
            post neonatal, then enter "early neonatal", "late neonatal",
            or "post neonatal" for age_start
            # TODO: Confirm with someone that this is the best way to get
            data for en, ln, and pn

    age_end: int or strt
        end point of the interpolation. all ages after this will have the
        same constant estimate of the quantity of interest that you are
        studying (see extrapolate_ages for more info)
            NOTE: If you want your age range to end with early, late, or
            post neonatal, then enter "early neonatal", "late neonatal",
            or "post neonatal" for age_end
            # TODO: Confirm with someone that this is the best way to get
            data for en, ln, and pn

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    measure : int, measure
        defines which measure (e.g. prevalence) you want to pull. Use central
        comp's get_ids functions to learn about which measures are available
        and what numbers correspond with each measure

    me_id: int, modelable entity id
        modelable_entity_id takes same me_id values as are used for GBD

    Returns
    -------
    df with year_id, sex_id, age and 1k draws. many of the draw columns have
    nulls at this point which will be filled in by interpolation function
    """

    # Set ages and years of interest
        # need to use special handling if age_start or age_end is early, late, 
        # or post neonatal
    
    # TODO: Use dbtools to grab en, ln, and pn midpts straight from the database
    en_midpt = (.01917808 / 2)
    ln_midpt = ((0.01917808 + 0.07671233) / 2)
    pn_midpt = ((0.07671233 + 1) / 2)
    
    # if age_start is <1 but age_end >1 
    if type(age_end) is int and type(age_start) is str:
        if age_start == "early neonatal":
            ages = [en_midpt, ln_midpt, pn_midpt] + np.arange(1, age_end + .5, .5).tolist()
        elif age_start == "late neonatal":
            ages = [ln_midpt, pn_midpt] + np.arange(1, age_end + .5, .5).tolist()
        elif age_start == "post neonatal":
            ages = [pn_midpt] + np.arange(1, age_end + .5, .5).tolist()
        else:
            raise ValueError("""you supplied an incorrect age_start value. you wrote {}
 which is invalid. valid entries for age_start are 'early neonatal', 'late neonatal',
 'post neonatal', or a number that is a multiple of .5 between 1 and 80 (inclusive)""".
format(age_start))

    # if age_start <1 and age_end <1
    elif type(age_end) and type(age_start) is str:        
        if age_start == "early neonatal" and age_end == "early neonatal":
            ages = [en_midpt] 
        if age_start == "early neonatal" and age_end == "late neonatal":
            ages = [en_midpt, ln_midpt]
        if age_start == "early neonatal" and age_end == "post neonatal":
            ages = [en_midpt, ln_midpt, pn_midpt]        
        elif age_start == "late neonatal" and age_end == "late neonatal":
            ages = [ln_midpt]
        elif age_start == "late neonatal" and age_end == "post neonatal":
            ages = [ln_midpt, pn_midpt]
        elif age_start == "post neonatal" and age_end =="post neonatal":
            ages = [pn_midpt]
        else:
            raise ValueError("""you supplied an incorrect age_start or age_end value.
 you wrote {s} and {e}, at least one of  which is invalid. valid entries for age_start 
 and age_end are early neonatal, late neonatal, post neonatal, or a number that is
 a multiple of .5 between 1 and 80 (inclusive)""".format(a=age_start, e=age_end))

    else:
        ages = np.arange(age_start, age_end + .5, .5).tolist()

    years = range(year_start, year_end + 1)

    # Set indexes of year_id and age
    indexed_df = df.set_index(['year_id', 'age']).sortlevel().copy()

    age_sex_index = pd.MultiIndex.from_product(
        [years, ages], names=['year_id', 'age'])

    expanded_data = pd.DataFrame(indexed_df, index=age_sex_index)

    if type(age_start) is int:
        assert age_start >= 1, """age_start cannot be an integer less than 1. if 
                                  you want age_start to be <1, state that it is
                                  equal to 'early neonatal', 'late neonatal', or
                                  'post neonatal'"""

    return expanded_data


def interpolate_linearly_over_years_then_ages(df, col_prefix1,
                                              col_prefix2=None):
    """
    Returns a dataframe with interpolated draw values

    Parameters
    ----------
    df: df
        df with columns year_id, sex_id, age, and 1k draws of a quantity of
        interest

    col_prefix: str
        prefix of the draw column that you will interpolate over (e.g. the
        col_prefix of 'rr_0' is 'rr')

    Returns
    -------
    df with year_id, sex_id, age and 1k draws. null values in the input file
    are now filled because we interpolated between the age/year combinations
    that we did have data for
    """
    if col_prefix1 == "angina_prop":
        keepcol = ["angina_prop"]

    else:
        keepcol = ["{c}_{i}".format(c=col_prefix1, i=i) for i in range(0, 1000)]

    if col_prefix2 is not None:
        keepcol += ["{c}_{i}".format(c=col_prefix2, i=i) for i in range(0, 1000)]

    interp_columns = df[keepcol]
    interp_data = interp_columns.groupby(level=0).apply(lambda x: x.interpolate())
    interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

    return interp_data


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


def get_age_from_age_group_id(df):
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
    # TODO: figure out what number to use for 80+ group
    mapping['age'] = mapping[['age_group_years_start', 'age_group_years_end']].mean(axis=1)
    df = df.set_index('age_group_id')
    df['age'] = mapping['age']
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

    aG, yG = np.meshgrid(a, y)  # create the actual grid
    aG = aG.flatten()  # make the grid 1d
    yG = yG.flatten()  # make the grid 1d
    return pd.DataFrame({'age': aG, 'year_id': yG})  # return a dataframe


def extrapolate_ages(df, age_end, year_start, year_end):
    """
    Extrapolates GBD data for simulants over the age of 80
    Necessary because GBD's estimates only go to "80+" and we
    need data for single ages greater than 80

    Parameters
    ----------
    df: dataframe of unextrapolated estimates
    age_end: age to which you want to extrapolate (inclusive)
    year_end: year to which you want to extrapolate (inclusive)

    Returns
    -------
    df with extrapolated values
    """

    expand_ages = range(df.age.max() + 1, age_end + 1)
    expand_years = range(year_start, year_end + 1)

    # use expand_grid auxilliary function to create a table
    # of expanded ages and years
    expand_table = expand_grid(expand_ages, expand_years)

    dup_table = df.query("age == {}".format(df.age.max()))

    # Do this only for the 80 plus year olds so that we can extend our
    # cause-deleted mortality rates to the older ages
    dup_table.reset_index(level=0, inplace=True)

    merged = pd.merge(expand_table, dup_table, on=['year_id'])

    df.reset_index(level=0, inplace=True)
    df.reset_index(level=1, inplace=True)

    df = df.append(merged)

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
    pop = stata_wrapper('get_populations.do', 'pop_{l}.csv'.format(l = location_id), location_id)

    # use auxilliary function extract_age_from_age_group_name to create an age
    # column
    pop = get_age_from_age_group_id(pop)

    # Grab population for year_start only (to initialize microsim population)
    pop = pop.query('year_id=={y}'.format(y=year_start))

    # Determine gender of interest. Can be 1, 2, or 3
    pop = pop.query("sex_id == {s}".format(s=sex_id))

    # Keep only the relevant columns
    pop = pop[['year_id', 'location_name',
               'location_id', 'age', 'sex_id', 'pop_scaled']]

    # assert an error if there are duplicate rows
    assert pop.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_populations just tried to output. check the population file that you pulled in from the GBD database"

    # assert an error to make sure data is dense (i.e. no missing data)
    assert pop.isnull().values.any() == False, "there are nulls in the dataframe that get_populations just tried to output. check the population file that you pulled in from the GBD database"

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

    # change sex_id column name to be an integer
    new_sim_file.sex_id = new_sim_file.sex_id.astype(int)

    # assert an error that only sex_id 1 and 2 are in the new_sim_file
    assert new_sim_file.sex_id.isin([1, 2]).all() == True, "something went wrong with assign_sex_id. function tried to assign a sex id other than 1 or 2"

    return new_sim_file


def get_all_cause_mortality_rate(location_id, year_start, year_end):
    # FIXME: for future models, actually bring in cause-deleted mortality
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

    all_cause_mr_dict = {}

    for sex_id in (1, 2):
        all_cause_mr = stata_wrapper('get_all_cause_mortality_rate_draws.do', 'all_cause_mortality_causeid294_in_country{l}.csv'.format(l = location_id), location_id)

        # filter so that only metric id 1 (deaths) is in our dataframe
        all_cause_deaths = all_cause_mr.query("metric_id == 1").copy()

        # read in and merge the population file to file with all_cause deaths to calculate the rate
        # rate = # of all cause deaths / population for every age, sex, year
        # combination
        pop = stata_wrapper('get_populations.do', 'pop_{l}.csv'.format(l = location_id), location_id)

        # merge all cause deaths and pop to get all cause mortality rate
        all_cause_mr = pd.merge(all_cause_deaths, pop, on=[
                                'age_group_id', 'year_id', 'sex_id', 'location_id'])

        # Need to divide # of all cause deaths by population
        for i in range(0, 1000):
            all_cause_mr['all_cause_mortality_rate_{}'.format(i)] = all_cause_mr['draw_{}'.format(i)]\
                / all_cause_mr.pop_scaled

        all_cause_mr = get_age_from_age_group_id(all_cause_mr).copy()

        # only get years we care about
        all_cause_mr = all_cause_mr.query('year_id>={ys} and year_id<={ye}'.
                                          format(ys=year_start, ye=year_end))

        all_cause_mr = all_cause_mr.query('sex_id == {s}'.format(s=sex_id))

        all_cause_mr = set_age_year_index(all_cause_mr, 'early neonatal',
                                          80, year_start, year_end)

        interp_data = interpolate_linearly_over_years_then_ages(all_cause_mr,
                                                                'all_cause_mortality_rate')

        interp_data['sex_id'] = sex_id

        all_cause_mr_dict[sex_id] = extrapolate_ages(
            interp_data, 105, year_start, year_end)

    output_df = all_cause_mr_dict[1].append(all_cause_mr_dict[2])

    output_df['location_id'] = location_id

    output_df.location_id = output_df.location_id.astype(int)

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_all_cause_mortality just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_all_cause_mortality_rate just tried to output. check the cache to make sure that the data you're pulling is correct"

    # assert that non of the all-cause mortality rate values are greater than 1
    draw_number = config.getint('run_configuration', 'draw_number')
    assert output_df['all_cause_mortality_rate_{}'.format(draw_number)].all() <= 1, "something went wrong with the get_all_cause_mortality_rate calculation. all-cause mortality can't be GT 0"

    return output_df


# TODO: write healthstate functions tests
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
