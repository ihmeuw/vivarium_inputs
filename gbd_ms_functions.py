# ~/ceam/ceam/gbd_data/gbd_ms_functions.py
# coding: utf-8


import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import choice
import os.path
from hashlib import md5
import os
import pdb

from ceam import config
from .gbd_ms_auxiliary_functions import normalize_for_simulation, get_age_from_age_group_id 
from .gbd_ms_auxiliary_functions import expand_grid, extrapolate_ages, get_populations
from .gbd_ms_auxiliary_functions import assign_sex_id, get_all_cause_mortality_rate

import logging
_log = logging.getLogger(__name__)

# cache_path = config.getstr('input_data', 'intermediary_data_cache_path') # TODO: After closing CE-241, we should be able to use this line
									    # until then, the central function scripts put all data in /share/costeffectiveness/CEAM/cache
cache_path = "/share/costeffectiveness/CEAM/cache/"

# # Microsim functions
# This notebook contains the functions that will be used to 
# re-format GBD data into a format that can be used for the cost-effectiveness microsim.
# Wherever possible, these functions will leverage the existing central comp functions 
# (please see this link for more information on the central computation functions
# https://hub.ihme.washington.edu/display/G2/Central+Function+Documentation)

# ### 1. generate_ceam_population
# # TODO: Figure out if we can assign ages at 5 year intervals


def generate_ceam_population(location_id, year_start, number_of_simulants):
    """
    Returns a population of simulants to be fed into CEAM

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
    """

    # Use auxilliary get_populations function to bring in the both sex
    # population
    pop = get_populations(location_id, year_start, 3)

    total_pop_value = pop.sum()['pop_scaled']

    # get proportion of total population in each age group
    pop['proportion_of_total_pop'] = pop['pop_scaled'] / total_pop_value

    # create a dataframe of 50k simulants
    simulants = pd.DataFrame({'simulant_id': range(0, number_of_simulants)})

    # use stats package to assign ages to simulants according to proportions in the
    # population file
    # TODO: use np.random.choice and assign age/sex at the same time
    ages = pop.age.values
    proportions = pop.proportion_of_total_pop.values
    simulant_ages = stats.rv_discrete(values=(ages, proportions))
    simulants['age'] = simulant_ages.rvs(size=number_of_simulants)

    simulants = assign_sex_id(simulants, location_id, year_start)
    
    # TODO: Design and implement test that makes sure CEAM population looks like population file pulled from GBD
    # TODO: Design and implement test that makes sure population has been smoothed out-- JIRA TIC CE-213
    
    # assert an error to make sure data is dense (i.e. no missing data)
    assert simulants.isnull().values.any() == False, "there are nulls in the dataframe that generate_ceam_population just tried to output. check the function and its auxiliary functions (get_populations and assign_sex_id)"

    # assert an error if there are duplicate rows
    assert simulants.duplicated(['simulant_id']).sum(
    ) == 0, "there are duplicates in the dataframe that generate_ceam_population just tried to output. check the function and its auxiliary functions (get_populations and assign_sex_id)"

    return simulants


# 2. assign_cause_at_beginning_of_simulation

def assign_cause_at_beginning_of_simulation(simulants_df, location_id, year_start, states):
    """
    Function that assigns chronic ihd status to starting population of simulants
    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made earlier in the function

    location_id : int, location id
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

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

    draw_number = config.getint('run_configuration', 'draw_number')
    keepcol = ['year_id', 'sex_id', 'age', 'draw_{}'.format(draw_number)]
    prevalence_draws_dictionary = {}

    for key, value in states.items():
        prevalence_draws_dictionary[key] = get_modelable_entity_draws(location_id,
                                                                      year_start, year_start, 5, value)
        prevalence_draws_dictionary[
            key] = prevalence_draws_dictionary[key][keepcol]
        prevalence_df = prevalence_df.append(prevalence_draws_dictionary[key])

    prevalence_df = prevalence_df.groupby(
        ['year_id', 'sex_id', 'age'], as_index=False).sum()

    for sex_id in simulants_df.sex_id.unique():
        for age in simulants_df.age.unique():
            elements = [0, 1]
            probability_of_disease = prevalence_df.\
                query("age=={a} and sex_id=={s}".format(a=age, s=sex_id))[
                    'draw_{}'.format(draw_number)]
            probability_of_NOT_having_disease = 1 - probability_of_disease
            weights = [float(probability_of_NOT_having_disease),
                       float(probability_of_disease)]

            one_age = simulants_df.query(
                "age=={a} and sex_id=={s}".format(a=age, s=sex_id)).copy()
            one_age['condition_envelope'] = one_age['age'].map(
                lambda x: choice(elements, p=weights))
            new_sim_file = new_sim_file.append(one_age)

    # produce a column of strings with nulls for healty, everyone else gets key value from states
    # add up to get total, divide each prevalence by total and then use that as the weights for np choice
    # need to ensure that only people with the cause can get a specific
    # sequelae

    # TODO: Should we be using groupby for these loops to ensure that we're not looping over an age/sex combo that
    # doesn't exist
    for key in states.keys():
        prevalence_draws_dictionary[key] = \
            pd.merge(prevalence_draws_dictionary[key], prevalence_df, on=['age', 'sex_id', 'year_id'],
                     suffixes=('_single', '_total'))
        single = prevalence_draws_dictionary[key]['draw_{}_single'.format(draw_number)]
        total = prevalence_draws_dictionary[key]['draw_{}_total'.format(draw_number)]
        prevalence_draws_dictionary[key]['scaled_prevalence'] = single / total

    for sex_id in new_sim_file.sex_id.unique():
        for age in new_sim_file.age.unique():
            list_of_weights = []
            for key, dataframe in states.items():
                weight_scale_prev_tuple = (key, prevalence_draws_dictionary[key].
                                           query("sex_id == {s} and age== {a}".format(s=sex_id, a=age))['scaled_prevalence'].values[0])
                list_of_weights.append(weight_scale_prev_tuple)

            list_of_keys, list_of_weights = zip(*list_of_weights)
            with_ihd = (new_sim_file.condition_envelope == 1) & (new_sim_file.age == age) & \
                       (new_sim_file.sex_id == sex_id)

            new_sim_file.loc[with_ihd, 'condition_state'] = np.random.choice(
                list_of_keys, p=list_of_weights, size=with_ihd.sum())

    new_sim_file = new_sim_file.apply(lambda x: x.fillna('healthy'), axis=0)

    # assert an error to make sure data is dense (i.e. no missing data)
    assert new_sim_file.isnull().values.any() == False, "there are nulls in the dataframe that assign_cause_at_beginning_of_simulation just tried to output. check that you've assigned the correct me_ids"
    
    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that assign_cause_at_beginning_of_simulation just tried to output. check that you've assigned the correct me_ids"

    return new_sim_file[['simulant_id', 'condition_state']]



ihd = [1814, 1817, 3233, 2412]
chronic_hemorrhagic_stroke = [9311, 9312]
list_of_me_ids_in_microsim = chronic_hemorrhagic_stroke + ihd



# ### 3. get_cause_deleted_mortality_rate

def get_cause_deleted_mortality_rate(location_id, year_start, year_end):
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
        csmr_draws = get_modelable_entity_draws(
            location_id, year_start, year_end, 15, me_id)
        all_me_id_draws = all_me_id_draws.append(csmr_draws)

    all_me_id_draws = all_me_id_draws.groupby(
        ['age', 'sex_id', 'year_id'], as_index=False).sum()

    all_cause_mr = get_all_cause_mortality_rate(
        location_id, year_start, year_end)

    cause_del_mr = pd.merge(all_cause_mr, all_me_id_draws, on=[
                            'age', 'sex_id', 'year_id'])

    for i in range(0, 1000):
        all_cause = cause_del_mr['all_cause_mortality_rate_{}'.format(i)]
        summed_csmr_of_sim_causes = cause_del_mr['draw_{}'.format(i)]
        cause_del_mr['cause_deleted_mortality_rate_{}'.format(i)] = all_cause - summed_csmr_of_sim_causes
    
    # assert an error to make sure data is dense (i.e. no missing data)
    assert cause_del_mr.isnull().values.any() == False, "there are nulls in the dataframe that get_cause_deleted_mortality_rate just tried to output. check the function as well as get_all_cause_mortality_rate"

    # assert an error if there are duplicate rows
    assert cause_del_mr.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_cause_deleted_mortality_rate just tried to output. check the function as well as get_all_cause_mortality_rate"

    # assert that non of the cause-deleted mortality rate values are less than or equal to 0
    draw_number = config.getint('run_configuration', 'draw_number')
    assert cause_del_mr['cause_deleted_mortality_rate_{}'.format(draw_number)].all() > 0, "something went wrong with the get_cause_deleted_mortality_rate calculation. all-cause mortality can't be <= 0"
   
    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('cause_deleted_mortality_rate_{i}'.format(i=i) for i in range(0, 1000)))

    return cause_del_mr[keepcol]


# ### 4. get_modelable_entity_draws (gives you incidence, prevalence, csmr, excess mortality, and other metrics at draw level)


def get_modelable_entity_draws(location_id, year_start, year_end, measure, me_id):
    """
    Returns draws for a given measure and modelable entity

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
    """

    output_df = pd.DataFrame()

    for sex_id in (1, 2):

        # assert an error to see if the data was pulled from the database
        assert os.path.isfile(cache_path + "draws_for_location{l}_for_meid{m}.csv".format(m=me_id, l=location_id)
        ) == True, "the draw info for me_id {m} in location_id {l} has not been pulled from the database or it is not in the correct place".format(m=me_id, l=location_id)
  
        draws = pd.read_csv(cache_path + "draws_for_location{l}_for_meid{m}.csv".format(m=me_id, l=location_id))

        draws = draws[draws.measure_id == measure]

        draws = draws.query('year_id>={ys} and year_id<={ye}'.format(
            ys=year_start, ye=year_end)).copy()

        draws = get_age_from_age_group_id(draws)

        draws = draws.query("sex_id == {s}".format(s=sex_id))

        # For now, do not include information on early, pre, and post neonatal
        draws = draws.query("age != 0")

        # Set ages and years of interest
        all_ages = range(1, 81)
        all_years = range(year_start, year_end + 1)

        # Set indexes of year_id and age
        draws = draws.set_index(['year_id', 'age']).sortlevel()

        ind = pd.MultiIndex.from_product(
            [all_years, all_ages], names=['year_id', 'age'])

        expanded_data = pd.DataFrame(draws, index=ind)

        # Keep only draw columns
        keepcol = ['draw_{i}'.format(i=i) for i in range(0, 1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(
            level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 151, year_start, year_end + 1))

        keepcol = ['year_id', 'sex_id', 'age']
        keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))
    
    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_modelable_entity_draws just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_modelable_entity_draws just tried to output. check the cache to make sure that the data you're pulling is correct"

    return output_df[keepcol].sort_values(by=['year_id', 'age', 'sex_id'])


# ### 5. get_heart_failure_incidence_draws

def get_heart_failure_incidence_draws(location_id, year_start, year_end, me_id):
    """
    Returns incidence draws for a given measure and cause of heart failure
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
    """

    # read in heart failure envelope. specify measure of interest
    hf_envelope = get_modelable_entity_draws(
        location_id, year_start, year_end, 6, 2412)

    # read in proportion of the cause of heart failure of interest
    proportion_draws = get_modelable_entity_draws(
        location_id, year_start, year_end, 18, me_id)

    # merge and then multiply envelope draws by proportion draws
    cause_of_hf = pd.merge(hf_envelope, proportion_draws, on=[
                           'age', 'year_id', 'sex_id'], suffixes=('_env', '_prop'))

    for i in range(0, 1000):
        envelope = cause_of_hf['draw_{i}_env'.format(i=i)]
        proportion = cause_of_hf['draw_{i}_prop'.format(i=i)]
        cause_of_hf['draw_{i}'.format(i=i)] = envelope * proportion

    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert cause_of_hf.isnull().values.any() == False, "there are nulls in the dataframe that get_heart_failure_incidence_draws just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert cause_of_hf.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_heart_failure_incidence_draws just tried to output. check the cache to make sure that the data you're pulling is correct"

    # assert that none of the incidence rate values are greater than 1 (basically ensuring that the numerator and demoniator weren't flipped)
    draw_number = config.getint('run_configuration', 'draw_number')
    assert cause_of_hf['draw_{}'.format(draw_number)].all() <= 1, "something went wrong with the get_heart_failure_incidence_draws calculation. incidence rate can't be GT 1. Check to see if the numerator/denominator were flipped"

    return cause_of_hf[keepcol]


# ### 6. get_relative_risks


def get_relative_risks(location_id, year_start, year_end, risk_id, cause_id):
    """
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
    """

    output_df = pd.DataFrame()

    for sex_id in (1, 2):

        # assert an error to see if the data was pulled from the database
        assert os.path.isfile(cache_path + "rel_risk_of_risk{r}_in_location{l}.csv".format(r=risk_id, l=location_id)
        ) == True, "the draw info for relative risk {r} in location_id {l} has not been pulled from the database or it is not in the correct place".format(r=risk_id, l=location_id)

        # Read in a csv of cause data that is produced by the get_outputs Stata
        # function
        RR = pd.read_csv(cache_path + "rel_risk_of_risk{r}_in_location{l}.csv".format(r=risk_id, l=location_id))

        RR = get_age_from_age_group_id(RR)

        RR = RR.query("cause_id == {c}".format(c=cause_id))

        RR = RR.query("sex_id == {s}".format(s=sex_id))

        RR = RR.query("age != 0")

        # need to treat risks with category parameters specially
        if risk_id == 166:
            RR = RR.query("parameter == 'cat1'")

        # Set ages and years of interest
        all_ages = range(1, 81)
        all_years = range(year_start, year_end + 1)

        # Set indexes of year_id and age
        RR = RR.set_index(['year_id', 'age']).sortlevel()

        ind = pd.MultiIndex.from_product(
            [all_years, all_ages], names=['year_id', 'age'])

        expanded_data = pd.DataFrame(RR, index=ind)

        # Keep only draw columns
        keepcol = ['rr_{i}'.format(i=i) for i in range(0, 1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(
            level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 151, year_start, year_end + 1))

        # need to back calculate relative risk to earlier ages for risks that don't start
        # until a certain age
        output_df = output_df.apply(lambda x: x.fillna(1), axis=0)

        keepcol = ['year_id', 'sex_id', 'age']
        keepcol.extend(('rr_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_relative_risks just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_relative_risks just tried to output. check the cache to make sure that the data you're pulling is correct"

    # assert that none of the rr values are less than 1
    draw_number = config.getint('run_configuration', 'draw_number')
    assert output_df['rr_{}'.format(draw_number)].all() >= 1, "something went wrong with get_relative_risks. RR cannot be LT 1. Check the data that you are pulling in and the function. Sometimes, the database does not have\
RR estimates for every age, so check to see that the function is correctly assigning relative risks to the other ages"
    
    return output_df[keepcol]


# ### 7. get_pafs

def get_pafs(location_id, year_start, year_end, risk_id, cause_id):
    """
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

    -------
    Returns
        df with columns year_id, sex_id, age, val, upper, and lower

    """

    output_df = pd.DataFrame()

    for sex_id in (1, 2):

        # assert an error to see if the data was pulled from the database
        assert os.path.isfile(cache_path + "PAFs_for_{c}_in_{l}.csv".format(c=cause_id, l=location_id)
        ) == True, "the pafs info for risk id {r} and cause id {c} in location_id {l} has not been pulled from the database or it is not in the correct place\
it is also possible that you are trying to pull a risk/cause combination that does not exist in GBD".format(r=risk_id, l=location_id, c=cause_id)

        pafs = pd.read_csv(cache_path + "PAFs_for_{c}_in_{l}.csv".\
                           format(c=cause_id, l=location_id))

        # only want metric id 2 (percentages or pafs)
        pafs = pafs.query("metric_id == 2")

        # only want one risk at a time
        pafs = pafs.query("rei_id == {r}".format(r=risk_id))

        pafs = get_age_from_age_group_id(pafs)

        pafs = pafs.query("sex_id == {s}".format(s=sex_id))

        all_ages = range(1, 81)
        all_years = range(year_start, year_end + 1)

        # Set indexes of year_id and age
        pafs = pafs.set_index(['year_id', 'age']).sortlevel()

        ind = pd.MultiIndex.from_product(
            [all_years, all_ages], names=['year_id', 'age'])

        expanded_data = pd.DataFrame(pafs, index=ind)

        keepcol = ['draw_{i}'.format(i=i) for i in range(0, 1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(
            level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 151, year_start, year_end + 1))

        # need to back calculate PAFS to earlier ages for risks that don't start
        # until a certain age
        output_df = output_df.apply(lambda x: x.fillna(0), axis=0)

        keepcol = ['year_id', 'sex_id', 'age']
        keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_pafs just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_pafs just tried to output. check the cache to make sure that the data you're pulling is correct"

    # assert that none of the paf values are greater than 1
    draw_number = config.getint('run_configuration', 'draw_number')
    assert output_df['draw_{}'.format(draw_number)].all() <= 1, "something went wrong with get_pafs. pafs cannot be GT 1. Check the data that you are pulling in and the function. Sometimes, the database does not have\
paf estimates for every age, so check to see that the function is correctly assigning relative risks to the other ages"

    return output_df[keepcol]


# 8. get_exposures


def get_exposures(location_id, year_start, year_end, risk_id):
    """
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

    Returns
    -------
    df with columns year_id, sex_id, age and 1k exposure draws
    """

    output_df = pd.DataFrame()

    for sex_id in (1, 2):
        
        # assert an error to see if the data was pulled from the database
        assert os.path.isfile(cache_path + "Exposure_of_risk{r}_in_location{l}.csv".format(r=risk_id, l=location_id)
        ) == True, "the exposure info for risk id {r} in location_id {l} has not been pulled from the database or it is not in the correct place".format(r=risk_id, l=location_id)

        exposure = pd.read_csv(cache_path + "Exposure_of_risk{r}_in_location{l}.csv".format(r=risk_id, l=location_id))

        exposure = get_age_from_age_group_id(exposure)

        exposure = exposure.query("sex_id == {s}".format(s=sex_id))

        exposure = exposure.query("age != 0")

        # need to treat risks with category parameters specially
        if risk_id == 166:
            exposure = exposure.query("parameter == 'cat1'")

        # Set ages and years of interest
        all_ages = range(1, 81)
        all_years = range(year_start, year_end + 1)

        # Set indexes of year_id and age
        exposure = exposure.set_index(['year_id', 'age']).sortlevel()

        ind = pd.MultiIndex.from_product(
            [all_years, all_ages], names=['year_id', 'age'])

        expanded_data = pd.DataFrame(exposure, index=ind)

        # Keep only draw columns
        keepcol = ['draw_{i}'.format(i=i) for i in range(0, 1000)]
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(
            level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 151, year_start, year_end + 1))

        keepcol += ['year_id', 'sex_id', 'age']

        output_df = output_df.apply(lambda x: x.fillna(0), axis=0)

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_exposures just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_relative_risks just tried to output. check the cache to make sure that the data you're pulling is correct"
    
    return output_df[keepcol]


# ### 9. TMREDs
# # TODO: Confirm that TMREDs are being calculated correct

# tmred_df = pd.read_excel('/snfs1/Project/Cost_Effectiveness/dev/data/gbd/risk_data/risk_variables.xlsx')

# # theoretical minimum risk exposure levels
# tmred_df = pd.read_excel('/snfs1/Project/Cost_Effectiveness/dev/data/gbd/risk_data/risk_variables.xlsx')

# # dictionary to hold TMREDs
# risk_tmred = {}

# # save max and min TMREDs to dictionary (distributions are all uniform)
# for risk in ['metab_sbp','smoking']:
#     risk_tmred[risk] = tmred_df.loc[tmred_df.risk==risk,['tmred_dist','tmred_para1','tmred_para2','rr_scalar','inv_exp']]

# risk_tmred['metab_sbp']

# risk_tmrel = {}

# # draw from uniform distribution for each risk factor
# for risk in ['metab_sbp']:
#     risk_tmrel[risk] = np.random.uniform(low=risk_tmred[risk]['tmred_para1'],high=risk_tmred[risk]['tmred_para2'],size=1)[0]
#     risk_tmrel[risk] = ((risk_tmred[risk]['tmred_para1'].values.astype(float)
#                          + risk_tmred[risk]['tmred_para2'].values.astype(float))/2)[0]

# risk_tmrel['metab_sbp']


# ### 10. load_data_from_cache

from joblib import Memory
memory = Memory(cachedir=config.get(
    'input_data', 'intermediary_data_cache_path'), verbose=1)


@memory.cache
def _inner_cached_call(funct, *args, **kwargs):
    return funct(*args, **kwargs)


def load_data_from_cache(funct, col_name, *args, **kwargs):
    """
    load_data_from_cache is a functor that will
    check a cache to see if data exists in that cache.
    If the data does not exist in the cache,
    load_data_from_cache will run a function (funct)
    with arguments (args,kwargs)

    Parameters
    ----------
    funct : str
        function to run if data is not already loaded into the cache
        (e.g. get_cause_deleted_mortality_rate)

    col_name : str
        rename the draw column to whichever column_name you want

    args,kwargs : int
        input the arguments required by the function (funct)
        (e.g. location_id, year_start, year_end)

    Returns
    -------
    df with input data for CEAM
    """

    # This causes the files that the cache writes to be both readable and
    # writeable by other users
    old_umask = os.umask(0)

    function_output = _inner_cached_call(funct, *args, **kwargs)

    os.umask(old_umask)

    if col_name:
        keepcol = ['year_id', 'age', 'sex_id', 'draw_{i}'.format(
            i=config.getint('run_configuration', 'draw_number'))]

        function_output = function_output[keepcol]
        function_output = function_output.rename(columns={'draw_{i}'.format(
            i=config.getint('run_configuration', 'draw_number')): col_name})

        return normalize_for_simulation(function_output)
    return function_output


# ### 11. get_severity_splits


# ### 12. get_sbp_mean_sd

def get_sbp_mean_sd(location_id, year_start, year_end):
    ''' Returns a dataframe of mean and sd of sbp in LOG SPACE

    Parameters
    ----------
    location_id : int

    year_start : int

    year_end : int

    Returns
    -------
    df with mean and sd values in LOG space
    '''
    output_df = pd.DataFrame()

    for sex_id in [1, 2]:
        draws = pd.DataFrame()
        for year_id in np.arange(year_start, year_end + 1, 5):

            # assert an error to see if the data was pulled from the database
            assert os.path.isfile("/share/epi/risk/paf/metab_sbp_interm/exp_{l}_{y}_{s}.dta".\
                                          format(l=location_id, y=year_id, s=sex_id)
            ) == True, "the sbp distribution files for location_id {l} do not seem to exist. we have had issues with pulling distribution data for some countries.\
if the data is truly not in the file -- /share/epi/risk/paf/metab_sbp_interm/-- then reach out to central comp to ask them to produce the data".format(r=risk_id, l=location_id)
            

            one_year_file = pd.read_stata("/share/epi/risk/paf/metab_sbp_interm/exp_{l}_{y}_{s}.dta".\
                                          format(l=location_id, y=year_id, s=sex_id))
            one_year_file['year_id'] = year_id
            draws = draws.append(one_year_file)

        draws = get_age_from_age_group_id(draws)

        # Set ages and years of interest
        all_ages = range(25, 81)
        all_years = range(year_start, year_end + 1)

        # Set indexes of year_id and age
        draws = draws.set_index(['year_id', 'age']).sortlevel()

        ind = pd.MultiIndex.from_product(
            [all_years, all_ages], names=['year_id', 'age'])

        expanded_data = pd.DataFrame(draws, index=ind)

        # Keep only mean and sd columns
        keepcol = ['exp_mean_{i}'.format(i=i) for i in range(0, 1000)]
        sdcol = ['exp_sd_{i}'.format(i=i) for i in range(0, 1000)]
        keepcol.extend(sdcol)
        mx = expanded_data[keepcol]

        # Interpolate over age and year
        interp_data = mx.groupby(level=0).apply(lambda x: x.interpolate())
        interp_data = interp_data.groupby(
            level=1).apply(lambda x: x.interpolate())

        interp_data['sex_id'] = sex_id

        for i in range(0,1000):
            exp_mean = interp_data['exp_mean_{}'.format(i)]
            exp_sd = interp_data['exp_sd_{}'.format(i)]
            interp_data['log_mean_{}'.format(i)] = np.log(
                exp_mean)
            interp_data['log_sd_{}'.format(i)] = (exp_sd / exp_mean)
                              
        output_df = output_df.append(
            extrapolate_ages(interp_data, 151, year_start, year_end + 1))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_sbp_mean_sd just tried to output. make sure what youre pulling from /share/epi/risk/paf/metab_sbp_interm/ is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_sbp_mean_sd just tried to output. make sure what youre pulling from /share/epi/risk/paf/metab_sbp_interm/ is correct"
    
    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('log_mean_{i}'.format(i=i) for i in range(0, 1000)))
    keepcol.extend(('log_sd_{i}'.format(i=i) for i in range(0, 1000)))

    return output_df[keepcol].sort_values(by=['year_id', 'age', 'sex_id'])


# End.
