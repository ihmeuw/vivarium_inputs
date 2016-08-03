
# coding: utf-8

# In[129]:

import numpy as np, pandas as pd
from scipy import stats
from numpy.random import choice
import os.path
from hashlib import md5
from ceam import config

import logging
_log = logging.getLogger(__name__)


# # Microsim functions
# This notebook contains version 2.0 of the functions that will be used to re-format GBD data into a format that can be used for the cost-effectiveness microsim. Wherever possible, these functions will leverage the existing central comp functions (please see this link for more information on the central computation functions https://hub.ihme.washington.edu/display/G2/Central+Function+Documentation)

# # Section 1 - Define auxilliary functions
# These functions will be used below to help simplify the code that actually reads in and re-formats GBD data.

# In[4]:

def normalize_for_simulation(df):
    # Convert sex_id to a categorical
    df['sex'] = df.sex_id.map({1:'Male', 2:'Female', 3:'Both'}).astype('category')
    df = df.drop('sex_id', axis=1)

    df = df.rename(columns={'year_id': 'year'})
    return df

def get_age_group_midpoint__from_age_group_name(df):
    """Creates an "age" column from the "age_group_name" column
    Age column values are ages at the midpoint of the age groups

    Parameters
    ----------
    df: dataframe for which you want an age column that has an age_group_id column
 
    Returns
    -------
    Midpoint of the age group values (e.g. age_group_name "5 to 9" returns 7.5
        TODO: Need to check with Haidong or someone else on mortality to see if
        we want to use age 82.5 for the 80+ age group
    """

    df['age'] = df['age_group_name'].map({})


# In[5]:

def get_age_from_age_group_id(df):
   """Creates an "age" column from the "age_group_id" column

   Parameters
   ----------
   df: df for which you want an age column that has an age_group_id column

   Returns
   -------
   df with an age column
   """

   df['age'] = df['age_group_id'].map({2: 0, 3: 0, 4: 0, 5: 1, 6: 5, 7: 10, 8: 15,
   9: 20, 10: 25, 11: 30, 12: 35, 13: 40, 14: 45, 15: 50, 16: 55, 17: 60, 18: 65,
   19: 70, 20: 75, 21: 80})

   return df


# In[6]:

def expand_grid(a, y):
    """ Creates an expanded dataframe of ages and years
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

    aG, yG = np.meshgrid(a, y) # create the actual grid
    aG = aG.flatten() # make the grid 1d
    yG = yG.flatten() # make the grid 1d
    return pd.DataFrame({'age':aG, 'year_id':yG}) # return a dataframe


# In[7]:

def extrapolate_ages(df,age_end,year_end):
    """Extrapolates GBD data for simulants over the age of 80
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

    TODO: Need to develop more sophisticated ways of extrapolating
          to higher ages and need to extrapolate farther into the
          future (currently doesn't forecast into the future and
          just uses data for 80 year olds for all ages GT 80)
    """

    expand_ages = range(81,age_end + 1)
    expand_years = range(1990,year_end + 1)

    # use expand_grid auxilliary function to create a table
    # of expanded ages and years
    expand_table = expand_grid(expand_ages,expand_years)

    dup_table = df.query("age == 80")

    # Do this only for the 80 plus year olds so that we can extend our
    # cause-deleted mortality rates to the older ages
    dup_table.reset_index(level=0, inplace=True)

    merged = pd.merge(expand_table, dup_table , on=['year_id'])

    df.reset_index(level=0, inplace=True)
    df.reset_index(level=1, inplace=True)

    df = df.append(merged)

    return df


# In[8]:

def get_populations(location_id,year_start,sex_id):

    """Get age-/sex-specific population structure

    Parameters
    ----------
    location_id : int, location id
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulatioN

    sex_id: str, sex
        sex_id takes values 1, 2, or 3

    Returns
    -------
    df with columns year_id, location_name, location_id, age, sex_id, and pop_scaled
      pop_scaled is the population for a given age/year/sex
    """

    # Read in a csv of population data that is produced by the get_populations Stata function
    pop = pd.read_csv("/share/costeffectiveness/CEAM/cache/pop_{l}.csv"                      .format(l = location_id))

    # assert an error to see if the population data was pulled from the database
    assert os.path.isfile("/share/costeffectiveness/CEAM/cache/pop_{l}.csv"                      .format(l = location_id)) == True, "the population information for location_id {l} has not been pulled from the database or it is not in the correct place".    format(l=location_id)

    # use auxilliary function extract_age_from_age_group_name to create an age column
    pop['age'] = pop['age_group_name'].map(extract_age_from_age_group_name)

    # Grab population for year_start only (to initialize microsim population)
    pop = pop.query('year_id=={y}'.format(y=year_start))

    # Determine gender of interest. Can be 1, 2, or 3
    pop = pop.query("sex_id == {s}".format(s = sex_id))

    # For now, don't include population for early, pre, post neonates
    pop = pop.query("age != 0") # TODO: Bring in EN, NN, PN eventually

    # Keep only the relevant columns
    keepcol=['year_id','location_name','location_id','age','sex_id','pop_scaled']

    pop = pop[keepcol]

    # assert an error if there are duplicate rows
    assert pop.duplicated(['age','year_id','sex_id']).sum() == 0,     "there are duplicates in the dataframe that get_populations just tried to output. check the population file that you pulled in from the GBD database"

    # assert an error to make sure data is dense (i.e. no missing data)
    assert pop.isnull().values.any() == False,     "there are nulls in the dataframe that get_populations just tried to output. check the population file that you pulled in from the GBD database"

    # Return a dataframe
    return pop


# In[9]

def assign_sex_id(simulants_df,location_id, year_start):
    """Assigns sex to a population of simulants so that age and sex are correlated

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

    new_sim_file = pd.DataFrame()

    # pull in male and female populations so that we can assign sex according to
    # GBD population estimates (with age/sex correlation)
    male_pop = get_populations(location_id,year_start,1)
    female_pop = get_populations(location_id,year_start,2)

    # do for each age in population dataframes (same ages in male_pop and female_pop)
    for age in male_pop.age.values:
        male_pop_value = male_pop.query("age=={a}".format(a=age)).pop_scaled.values
        female_pop_value = female_pop.query("age=={a}".format(a=age)).pop_scaled.values

        elements = [1,2]
        male_prop = male_pop_value / (male_pop_value + female_pop_value)
        female_prop = 1 - male_prop
        weights = [float(male_prop),float(female_prop)]

        one_age = simulants_df.query("age=={a}".format(a=age))
        one_age['sex_id'] = one_age['age'].map(lambda x: choice(elements, p=weights))

        new_sim_file = new_sim_file.append(one_age)

    return new_sim_file


# In[10]:

def assign_ihd(simulants_df, location_id, year_start):
    """Function that assigns chronic ihd status to starting population of simulants

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
    Creates a new column for a df of simulants with a column called chronic_ihd
        chronic_ihd takes values 0 or 1
            1 indicates that the simulant has chronic ihd
            0 indicates that the simulant does not have chronic ihd
    """

    new_sim_file = pd.DataFrame()

    angina_prevalence_draws = get_modelable_entity_draws(location_id,year_start,year_start,5,1817)
    asympt_prevalence_draws = get_modelable_entity_draws(location_id,year_start,year_start,5,3233)
    hf_ihd_prevalence_draws = get_modelable_entity_draws(location_id,year_start,year_start,5,9567)

    for i in range(0,1000):
        angina_prevalence_draws = angina_prevalence_draws.        rename(columns={'draw_{i}'.format(i=i):'angina_prev{i}'.format(i=i)})

        hf_ihd_prevalence_draws = hf_ihd_prevalence_draws.        rename(columns={'draw_{i}'.format(i=i):'hf_prev{i}'.format(i=i)})

        asympt_prevalence_draws = asympt_prevalence_draws.        rename(columns={'draw_{i}'.format(i=i):'asympt_prev{i}'.format(i=i)})

    prevalence_draws = angina_prevalence_draws.merge(hf_ihd_prevalence_draws, on = ['year_id','sex_id','age']).    merge(asympt_prevalence_draws, on = ['year_id','sex_id','age'])

    for i in range(0,1000):
        prevalence_draws['estimate_{i}'.format(i=i)] =         prevalence_draws['hf_prev{i}'.format(i=i)] +         prevalence_draws['asympt_prev{i}'.format(i=i)] +         prevalence_draws['angina_prev{i}'.format(i=i)]

    #TODO: Confirm if we want to use draws (as opposed to mean of draws) moving forward
    # If we use draws, we'll need to create the population file for every draw (1k times)
    prevalence_draws['mean'] = prevalence_draws.loc[:,'estimate_0':'estimate_999'].mean(axis=1)

    for sex_id in simulants_df.sex_id.unique():
        for age in simulants_df.age.unique():
            elements = [0,1]
            probability_of_disease = prevalence_draws.            query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))['mean']
            probability_of_NOT_having_disease = 1 - probability_of_disease
            weights = [float(probability_of_NOT_having_disease),
                       float(probability_of_disease)]

            one_age = simulants_df.query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))
            one_age['chronic_ihd'] = one_age['age'].map(lambda x: choice(elements, p=weights))
            new_sim_file = new_sim_file.append(one_age)

    return new_sim_file


# In[11]:

def assign_disease(simulants_df, location_id, year_start, me_id, col_name):
    """Function that assigns disease status to starting population of simulants

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    me_id : int
        modelable entity

    col_name : str
        name of the column that you want to create
        (i.e. name of the disease)

    Returns
    -------
    Creates a new column for a df of simulants with a column for the disease
    which you are assigning
        the column takes values 0 or 1
            1 indicates that the simulant has the disease
            0 indicates that the simulant does not have the disease
    """

    df = simulants_df

    new_sim_file = pd.DataFrame()

    prevalence_draws = get_modelable_entity_draws(location_id,year_start,year_start,5,me_id)

    #TODO: Confirm if we want to use draws (as opposed to mean of draws) moving forward
    # If we use draws, we'll need to create the population file for every draw (1k times)
    prevalence_draws['mean'] = prevalence_draws.loc[:,'draw_0':'draw_999'].mean(axis=1)

    for sex_id in simulants_df.sex_id.unique():
        for age in simulants_df.age.unique():
            elements = [0,1]
            probability_of_disease = prevalence_draws.            query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))['mean']
            probability_of_NOT_having_disease = 1 - probability_of_disease
            weights = [float(probability_of_NOT_having_disease),
                       float(probability_of_disease)]

            one_age = df.query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))
            one_age['{c}'.format(c=col_name)] = one_age['age'].map(lambda x: choice(elements, p=weights))
            new_sim_file = new_sim_file.append(one_age)

    return new_sim_file['{c}'.format(c=col_name)]


# In[12]:

def get_all_cause_mortality_rate(location_id, year_start, year_end):
    #FIXME: for future models, actually bring in cause-deleted mortality
    '''Get all-cause mortality rate from year_start to year_end (inclusive)

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

    for sex_id in (1,2):

        # Read in a csv of cause data that is produced by the get_outputs Stata function
        all_cause_mr = pd.read_csv("/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data/all_cause_mortality_causeid294_in_country{l}.csv".format(l = location_id))

        # only get years we care about and only get "Rate" rows, since we want the mortality rate
        all_cause_mr = all_cause_mr.query('year_id>={ys} and year_id<={ye}'.                                          format(ys=year_start, ye=year_end)).copy()

        # FIXME: Will want to use age midpoints in the future
        all_cause_mr['age'] = all_cause_mr['age_group_name'].map(extract_age_from_age_group_name)

        all_cause_mr = all_cause_mr.query('sex_id == {s}'.format(s=sex_id))

        # TODO: Figure out how to interpolate to the early, pre, and post neonatal groups
        all_cause_mr = all_cause_mr.query("age != 0")


        # create list of all ages/years we want
        all_ages = range(1,81)
        all_years = range(year_start,year_end + 1)

        # Set indexes on year_id and age
        all_cause_mr = all_cause_mr.set_index(['year_id','age']).sortlevel()

        ind = pd.MultiIndex.from_product([all_years,all_ages],names=['year_id','age'])

        expanded_data = pd.DataFrame(all_cause_mr,index=ind)

        # Keep only relevant columns
        keepcol=['val','upper','lower']
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

        # Rename the 'val' column to be more descriptive
        interp_data = interp_data.rename(columns={'val': 'all_cause_mortality_rate'})

        interp_data['sex_id'] = sex_id

        all_cause_mr_dict[sex_id] = extrapolate_ages(interp_data,151,year_end +1)

    output_df = all_cause_mr_dict[1].append(all_cause_mr_dict[2])

    output_df['location_id'] = location_id

    output_df.location_id = output_df.location_id.astype(int)

    return output_df

def assign_cause_at_beginning_of_simulation(simulants_df, location_id, year_start, col_name, states):
    
    """Function that assigns chronic ihd status to starting population of simulants
    
    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made earlier in the function
    
    location_id : int, location id
        location_id takes same location_id values as are used for GBD
    
    year_start : int, year
        year_start is the year in which you want to start the simulation
        
    seqlist : list
        list of modelable entity ids of the sequelae of a cause
        
    col_name : str
        name of the column that you want to create
        (i.e. name of the disease)
        
    states : dict
    	dict with keys = name of cause, values = modelable entity id of cause

    Returns
    -------
    Creates a new column for a df of simulants with a column called chronic_ihd
        chronic_ihd takes values 0 or 1
            1 indicates that the simulant has chronic ihd
            0 indicates that the simulant does not have chronic ihd
    """
    
    prevalence_df = pd.DataFrame()
    
    new_sim_file = pd.DataFrame()
    
    keepcol = ['year_id','sex_id','age','draw_0']
    prevalence_draws_dictionary = {}
    
    for key, value in states.items():
        prevalence_draws_dictionary[key] = get_modelable_entity_draws(location_id,
        year_start, year_start, 5, value)
        prevalence_draws_dictionary[key] = prevalence_draws_dictionary[key][keepcol]
        prevalence_df = prevalence_df.append(prevalence_draws_dictionary[key])

    prevalence_df = prevalence_df.groupby(['year_id','sex_id','age'], as_index=False).sum()
        
    for sex_id in simulants_df.sex_id.unique():
        for age in simulants_df.age.unique():
            elements = [0,1]
            probability_of_disease = prevalence_df.\
            query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))['draw_0']
            probability_of_NOT_having_disease = 1 - probability_of_disease
            weights = [float(probability_of_NOT_having_disease),
                       float(probability_of_disease)]

            one_age = simulants_df.query("age=={a} and sex_id=={s}".format(a=age,s=sex_id))
            one_age['{c}'.format(c=col_name)] = one_age['age'].map(lambda x: choice(elements, p=weights))
            new_sim_file = new_sim_file.append(one_age)
            
    # produce a column of strings with nulls for healty, everyone else gets key value from states
    # add up to get total, divide each prevalence by total and then use that as the weights for np choice
    # need to ensure that only people with the cause can get a specific sequelae
    
    # TODO: Should we be using groupby for these loops to ensure that we're not looping over an age/sex combo that
    # doesn't exist
    for key in states.keys():
        prevalence_draws_dictionary[key] = \
            pd.merge(prevalence_draws_dictionary[key], prevalence_df, on=['age','sex_id','year_id'], 
                suffixes=('_single', '_total'))
        prevalence_draws_dictionary[key]['scaled_prevalence'] = prevalence_draws_dictionary[key]['draw_0_single'] / \
            prevalence_draws_dictionary[key]['draw_0_total']

    for sex_id in new_sim_file.sex_id.unique():
        for age in new_sim_file.age.unique():
            list_of_weights = []
            for key, dataframe in states.items():
                weight_scale_prev_tuple =(key, prevalence_draws_dictionary[key].\
                                        query("sex_id == {s} and age== {a}".format(s=sex_id, a=age))['scaled_prevalence'].values[0])
                list_of_weights.append(weight_scale_prev_tuple)
                
            list_of_keys, list_of_weights = zip(*list_of_weights)
            with_ihd = (new_sim_file.ihd == 1) & (new_sim_file.age == age) & \
                       (new_sim_file.sex_id == sex_id)
        
            new_sim_file.loc[with_ihd, 'condition'] = np.random.choice(list_of_keys, p=list_of_weights, 
                                                                                      size=with_ihd.sum())
    
    return new_sim_file





# In[14]:

# def interpolate
# this is a placeholder. we'll eventually want an interpolation function but
# for now are content using Python's built in linear interpolation function


#TODO: Think of a more eloquent way to get all of the causes into a list
ihd = [1814,1817,3233]
chronic_hemorrhagic_stroke = [9311,9312]
list_of_me_ids_in_microsim = chronic_hemorrhagic_stroke + ihd


# # Section 4 - Define main functions
# These functions will be used to re-format GBD data that will be used in the cost-effectiveness microsim.

# ### 1. Generate a population of simulants with age, sex, and disease prevalence characteristics according to
# # TODO: Figure out if we can assign ages at 5 year intervals

# In[19]:

def generate_ceam_population(location_id,year_start,number_of_simulants):
    '''Returns a population of simulants to be fed into CEAM

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    number of simulants : int, number
        year_end is the year in which you want to end the simulation


    Returns
    -------
    df with columns simulant_id, age, sex_id, and columns to indicate if
    simulant has different diseases
    '''

    # Use auxilliary get_populations function to bring in the both sex population
    pop = get_populations(location_id,year_start,3)

    total_pop_value = pop.sum()['pop_scaled']

    # get proportion of total population in each age group
    pop['proportion_of_total_pop'] = pop['pop_scaled'] / total_pop_value

    # create a dataframe of 50k simulants
    simulants = pd.DataFrame({'simulant_id':range(0,number_of_simulants)})

    # use stats package to assign ages to simulants according to proportions in the
    # population file
    #TODO: use np.random.choice and assign age/sex at the same time
    ages = pop.age.values
    proportions = pop.proportion_of_total_pop.values
    simulant_ages = stats.rv_discrete(values=(ages, proportions))
    simulants['age'] = simulant_ages.rvs(size=number_of_simulants)

    simulants = assign_sex_id(simulants,location_id,year_start)
    
    return simulants


# ### 2. get cause-deleted mortality rate

# In[148]:

def get_cause_deleted_mortality_rate(location_id,year_start,year_end):
    '''Returns the cause-delted mortality rate for a given time period and location

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
    df with columns age, year_id, sex_id, and 1k draws of cause deleted mortality rate
    '''
    all_me_id_draws = pd.DataFrame()

    for me_id in list_of_me_ids_in_microsim:
        csmr_draws = get_modelable_entity_draws(location_id, year_start, year_end, 15, me_id) #15 is CSMR
        all_me_id_draws = all_me_id_draws.append(csmr_draws)

    all_me_id_draws = all_me_id_draws.groupby(['age', 'sex_id', 'year_id'], as_index=False).sum()

    all_cause_mr = get_all_cause_mortality_rate(location_id, year_start, year_end)

    cause_del_mr = pd.merge(all_cause_mr, all_me_id_draws, on=['age', 'sex_id', 'year_id'])

    keepcol = ['age','year_id','sex_id']
    for i in range(0,1000):
        cause_del_mr['draw_{i}'.format(i=i)] = cause_del_mr.all_cause_mortality_rate - cause_del_mr['draw_{i}'.format(i=i)]
        keepcol.append('draw_{i}'.format(i=i))

    return cause_del_mr[keepcol]


# ### 3. Get modelable entity draws (gives you incidence, prevalence, csmr, excess mortality, and other metrics at draw level)

# In[22]:

def get_modelable_entity_draws(location_id, year_start, year_end, measure, me_id):
    '''Returns draws for a given measure and modelable entity

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

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
    df with year_id, sex_id, age and 1k draws

    '''
    output_df = pd.DataFrame()

    for sex_id in (1,2):

        draws = pd.read_csv("/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data/draws_for_location{l}_for_meid{m}.csv".format(m=me_id, l=location_id))

        draws = draws[draws.measure_id == measure]

        draws = draws.query('year_id>={ys} and year_id<={ye}'.format(ys=year_start, ye=year_end)).copy()

        draws = get_age_from_age_group_id(draws)
 
        draws = draws.query("sex_id == {s}".format(s=sex_id))

        # For now, do not include information on early, pre, and post neonatal
        draws = draws.query("age != 0")

        # Set ages and years of interest
        all_ages = range(1,81) 
        all_years = range(year_start,year_end+1) 

        # Set indexes of year_id and age
        draws = draws.set_index(['year_id','age']).sortlevel()

        ind = pd.MultiIndex.from_product([all_years,all_ages],names=['year_id','age'])

        expanded_data = pd.DataFrame(draws,index=ind)

        # Keep only draw columns
        keepcol = ['draw_{i}'.format(i=i) for i in range(0,1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id']= sex_id

        output_df = output_df.append(extrapolate_ages(interp_data,151,year_end +1))

        keepcol = ['year_id','sex_id','age']
        keepcol.extend(('draw_{i}'.format(i=i) for i in range(0,1000)))

    return output_df[keepcol].sort_values(by=['year_id','age','sex_id'])


# ### 4. Get heart failure draws

# In[119]:

def get_heart_failure_incidence_draws(location_id,year_start, year_end, me_id):
    '''Returns incidence draws for a given measure and cause of heart failure
    Since GBD 2015 does not have full models for specific causes of heart failure,
    get_heart_failure_draws approximates full models through reading in data for
    the entire heart failure impairment envelope and then multipying the envelope
    by the proportion of hf due to specific causes

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int
        year_start is the year in which you want to start the simulation

    year_end : int
        year_end is the year in which you want to end the simulation

    me_id: int
        modelable_entity_id takes same me_id values as are used for GBD
        corresponds with the me_id of the cause of heart failure that is
        of interest

    Returns
    -------
    df with year_id, sex_id, age and 1k draws
    '''

    # read in heart failure envelope. specify measure of interest
    hf_envelope = get_modelable_entity_draws(location_id, year_start, year_end, 6, 2412)

    # read in proportion of the cause of heart failure of interest
    proportion_draws = get_modelable_entity_draws(location_id, year_start, year_end, 18, me_id)

    # merge and then multiply envelope draws by proportion draws
    cause_of_hf = pd.merge(hf_envelope, proportion_draws,on=['age','year_id','sex_id'], suffixes=('_env','_prop'))

    for i in range(0,1000):
        cause_of_hf['draw_{i}'.format(i=i)] =  cause_of_hf['draw_{i}_env'.format(i=i)] *         cause_of_hf['draw_{i}_prop'.format(i=i)]

    keepcol = ['year_id','sex_id','age']
    keepcol.extend(('draw_{i}'.format(i=i) for i in range(0,1000)))

    return cause_of_hf[keepcol]

# ### 5. Get Relative Risks

# In[24]:
def get_relative_risks(location_id,year_start,year_end,risk_id,cause_id):
    '''
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    risk_id: int, risk id
        risk_id takes same risk_id values as are used for GBD

    cause_id: int, cause id
        cause_id takes same cause_id values as are used for GBD

    Returns
    -------
    df with columns year_id, sex_id, age, 1k relative risk draws

    '''

    output_df = pd.DataFrame()

    for sex_id in (1,2):

        # Read in a csv of cause data that is produced by the get_outputs Stata function
        RR = pd.read_csv("/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data/rel_risk_of_risk{r}_in_location{l}.csv".format(r=risk_id,l=location_id))

        RR = get_age_from_age_group_id(RR)

        RR = RR.query("cause_id == {c}".format(c=cause_id))

        RR = RR.query("sex_id == {s}".format(s=sex_id))

        RR = RR.query("age != 0")

        # need to treat risks with category parameters specially
        if risk_id == 166:
            RR = RR.query("parameter == 'cat1'")

        # Set ages and years of interest
        all_ages = range(1,81)
        all_years = range(year_start,year_end+1)


        # Set indexes of year_id and age
        RR = RR.set_index(['year_id','age']).sortlevel()

        ind = pd.MultiIndex.from_product([all_years,all_ages],names=['year_id','age'])

        expanded_data = pd.DataFrame(RR,index=ind)

        # Keep only draw columns
        keepcol = ['rr_{i}'.format(i=i) for i in range(0,1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id']= sex_id

        output_df = output_df.append(extrapolate_ages(interp_data,151,year_end +1))

        # need to back calculate relative risk to earlier ages for risks that don't start
        # until a certain age
        output_df = output_df.apply(lambda x: x.fillna(1),axis = 0)

        keepcol = ['year_id','sex_id','age']
        keepcol.extend(('rr_{i}'.format(i=i) for i in range(0,1000)))

    return output_df[keepcol]


# ### 6. PAFs

# In[25]:

def get_pafs(location_id,year_start,year_end,risk_id,cause_id):
    '''
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    risk_id: int, risk id
        risk_id takes same risk_id values as are used for GBD

    cause_id: int, cause id
        cause_id takes same cause_id values as are used for GBD

    Returns/\

    -------
    df with columns year_id, sex_id, age, val, upper, and lower

    '''
    output_df = pd.DataFrame()

    for sex_id in (1,2):

        pafs = pd.read_csv("/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data/\
PAFs_of_risk_{r}_for_{c}_in_{l}.csv".format(r=risk_id,c=cause_id,l=location_id))

        # smoking pafs file contains values for non-smokers
        pafs = pafs.dropna()

        pafs['age'] = pafs['age_group_name'].map(extract_age_from_age_group_name)

        pafs = pafs.query("sex_id == {s}".format(s=sex_id))

        all_ages = range(1,81) #TODO: Figure out how to extrapolate
        all_years = range(year_start,year_end)

        # Set indexes of year_id and age
        pafs = pafs.set_index(['year_id','age']).sortlevel()

        ind = pd.MultiIndex.from_product([all_years,all_ages],names=['year_id','age'])

        expanded_data = pd.DataFrame(pafs,index=ind)

        keepcol = ['val','upper','lower']
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id']= sex_id

        output_df = output_df.append(extrapolate_ages(interp_data,151,year_end +1))

        # need to back calculate PAFS to earlier ages for risks that don't start
        # until a certain age
        output_df = output_df.apply(lambda x: x.fillna(0),axis = 0)

        keepcol = ['year_id','sex_id','age','val','upper','lower']

    return output_df[keepcol]


# ### 8. Exposures
# # TODO: Clarify what category 1 is for smoking

# In[152]:

def get_exposures(location_id,year_start,year_end,risk_id):
    '''
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    risk_id: int, risk id
        risk_id takes same risk_id values as are used for GBD

    Returns
    -------
    df with columns year_id, sex_id, age and 1k exposure draws

    '''

    output_df = pd.DataFrame()

    for sex_id in (1,2):

        exposure = make_exposure_draws_one_df(location_id,risk_id)

        exposure = get_age_from_age_group_id(exposure)

        exposure = exposure.query("sex_id == {s}".format(s=sex_id))

        exposure = exposure.query("age != 0")

        # need to treat risks with category parameters specially
        if risk_id == 166:
            exposure = exposure.query("parameter == 'cat1'")

        # Set ages and years of interest
        all_ages = range(exposure.age.min(),81)
        all_years = range(year_start,year_end+1)

        # Set indexes of year_id and age
        exposure = exposure.set_index(['year_id','age']).sortlevel()

        ind = pd.MultiIndex.from_product([all_years,all_ages],names=['year_id','age'])

        expanded_data = pd.DataFrame(exposure,index=ind)

        # Keep only draw columns
        keepcol = ['draw_{i}'.format(i=i) for i in range(0,1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id']= sex_id

        output_df = output_df.append(extrapolate_ages(interp_data,151,year_end +1))

        keepcol = ['year_id','sex_id','age']
        keepcol.extend(('draw_{i}'.format(i=config.getint('run_configuration', 'draw_number'))))

    return output_df


# ### 9. SBP Dist
# # TODO: Ask Reed what is in this file. Need a better name than SBP Dist

# ### 10. TMREDs
# # TODO: Confirm that TMREDs are being calculated correct

# In[27]:

# tmred_df = pd.read_excel('/snfs1/Project/Cost_Effectiveness/dev/data/gbd/risk_data/risk_variables.xlsx')


# In[28]:

# # theoretical minimum risk exposure levels
# tmred_df = pd.read_excel('/snfs1/Project/Cost_Effectiveness/dev/data/gbd/risk_data/risk_variables.xlsx')

# # dictionary to hold TMREDs
# risk_tmred = {}

# # save max and min TMREDs to dictionary (distributions are all uniform)
# for risk in ['metab_sbp','smoking']:
#     risk_tmred[risk] = tmred_df.loc[tmred_df.risk==risk,['tmred_dist','tmred_para1','tmred_para2','rr_scalar','inv_exp']]


# In[29]:

# risk_tmred['metab_sbp']


# In[30]:

# risk_tmrel = {}

# # draw from uniform distribution for each risk factor
# for risk in ['metab_sbp']:
#     risk_tmrel[risk] = np.random.uniform(low=risk_tmred[risk]['tmred_para1'],high=risk_tmred[risk]['tmred_para2'],size=1)[0]
#     risk_tmrel[risk] = ((risk_tmred[risk]['tmred_para1'].values.astype(float)
#                          + risk_tmred[risk]['tmred_para2'].values.astype(float))/2)[0]


# In[31]:

# risk_tmrel['metab_sbp']


# ### 11. Load data from cache

# In[147]:

def load_data_from_cache(funct, col_name, *args, **kwargs):
    '''load_data_from_cache is a functor that will
    check a cache to see if data exists in that cache.
    If the data does not exist in the cache,
    load_data_from_cache will run a function (funct)
    with arguments (args,kwargs)

    Parameters
    ----------
    funct : str
        function ro run if data is not already loaded into the cache
        (e.g. get_cause_deleted_mortality_rate)

    col_name : str
        rename the draw column to whichever column_name you want

    args,kwargs : int
        input the arguments required by the function (funct)
        (e.g. location_id, year_start, year_end)


    Returns
    -------
    df with input data for CEAM
    '''

    file_name = funct.__name__ + '_' + md5(str((args, sorted(kwargs.items()))).encode('utf-8')).hexdigest() + '.csv'

    path = os.path.join(config.get('input_data', 'intermediary_data_cache_path'), file_name)

    if os.path.exists(path):
        _log.debug('Loading %s from cache', (funct, args, kwargs))
        function_output = pd.read_csv(path)
    else:
        _log.debug('Loading %s into cache', (funct, args, kwargs))
        function_output = funct(*args, **kwargs)
        function_output.to_csv(path)

    keepcol = ['year_id','age','sex_id','draw_{i}'.format(i=config.getint('run_configuration', 'draw_number'))]

    function_output = function_output[keepcol]
    function_output = function_output.rename(columns={'draw_{i}'.format(i=config.getint('run_configuration', 'draw_number')) : col_name})

    return normalize_for_simulation(function_output)



# ### 12. Severity Splits


