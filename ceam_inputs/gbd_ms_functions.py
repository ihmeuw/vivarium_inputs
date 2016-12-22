# ~/ceam/ceam/gbd_data/gbd_ms_functions.py
# coding: utf-8

import os.path
import os
import shutil
from datetime import timedelta

import numpy as np
import pandas as pd

from scipy.stats import beta

from joblib import Memory
from flufl.lock import Lock

from db_tools import ezfuncs

from ceam import config
from ceam.interpolation import Interpolation
from ceam.framework.randomness import choice

from ceam_inputs.util import stata_wrapper, get_cache_directory
from ceam_inputs.auxiliary_files import open_auxiliary_file, auxiliary_file_path

from ceam_inputs.gbd_ms_auxiliary_functions import set_age_year_index
from ceam_inputs.gbd_ms_auxiliary_functions import interpolate_linearly_over_years_then_ages
from ceam_inputs.gbd_ms_auxiliary_functions import create_age_column
from ceam_inputs.gbd_ms_auxiliary_functions import normalize_for_simulation
from ceam_inputs.gbd_ms_auxiliary_functions import get_age_from_age_group_id
from ceam_inputs.gbd_ms_auxiliary_functions import expand_grid
from ceam_inputs.gbd_ms_auxiliary_functions import extrapolate_ages
from ceam_inputs.gbd_ms_auxiliary_functions import get_populations
from ceam_inputs.gbd_ms_auxiliary_functions import create_sex_id_column
from ceam_inputs.gbd_ms_auxiliary_functions import get_all_cause_mortality_rate
from ceam_inputs.gbd_ms_auxiliary_functions import get_healthstate_id
from ceam.interpolation import Interpolation
from ceam.framework.randomness import choice
# em 9/21: do we want to be converting from rates to probabilities in gbd_ms_functions.py?
# TODO: Yes bring in probabilities. BUT CONFIRM WITH ABIE THAT WE WANT TO BE USING ANNUAL RATES HERE
from joblib import Memory
import warnings


from ceam.framework.util import from_yearly, rate_to_probability

import logging
_log = logging.getLogger(__name__)

memory = Memory(cachedir=get_cache_directory(), verbose=1)


# # Microsim functions
# This notebook contains the functions that will be used to
# re-format GBD data into a format that can be used for the cost-effectiveness
# microsim. Wherever possible, these functions will leverage the existing
# central comp functions (please see this link for more information on the
# central computation functions
# https://hub.ihme.washington.edu/display/G2/Central+Function+Documentation)

@memory.cache
def get_model_versions():
    """Return a mapping from modelable_entity_id to the version of that entity 
    associated with the GBD publications currently configured.
    """
    publication_ids = [int(pid) for pid in config.get('input_data', 'gbd_publication_ids').split(',')]
    mapping = ezfuncs.query('''
    SELECT modelable_entity_id, model_version_id
    FROM epi.publication_model_version
    JOIN epi.model_version USING (model_version_id)
    JOIN shared.publication USING (publication_id)
    WHERE publication_id in ({})
    '''.format(','.join([str(pid) for pid in publication_ids]))
    , conn_def='epi')

    mapping = dict(mapping[['modelable_entity_id', 'model_version_id']].values)

    return mapping

# 1. get_modelable_entity_draws (gives you incidence, prevalence, csmr, excess mortality, and other metrics at draw level)


def get_modelable_entity_draws(location_id, year_start, year_end, measure,
                               me_id):
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
    meid_version_map = get_model_versions()
    model_version = meid_version_map[me_id]

    for sex_id in (1, 2):


        draws = stata_wrapper('get_modelable_entity_draws.do', 'draws_for_location{l}_for_meid{m}.csv'.format(m=me_id, l=location_id), location_id, me_id, model_version)

        draws = draws[draws.measure_id == measure]

        draws = draws.query('year_id>={ys} and year_id<={ye}'.format(
            ys=year_start, ye=year_end)).copy()

        draws = get_age_from_age_group_id(draws)

        draws = draws.query("sex_id == {s}".format(s=sex_id))

        draws = set_age_year_index(draws, 'early neonatal', 80, year_start, year_end)

        interp_data = interpolate_linearly_over_years_then_ages(draws, 'draw')

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 105, year_start, year_end))

        keepcol = ['year_id', 'sex_id', 'age']
        keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_modelable_entity_draws just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_modelable_entity_draws just tried to output. check the cache to make sure that the data you're pulling is correct"

    return output_df[keepcol].sort_values(by=['year_id', 'age', 'sex_id'])


# 2. generate_ceam_population
# TODO: Figure out if we can assign ages at 5 year intervals


def generate_ceam_population(location_id, year_start, number_of_simulants, initial_age=None):
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

    initial_age : int
        If not None simulants will all be set to this age otherwise their
        ages will come from the population distribution

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

    if initial_age is None:
        simulants = create_age_column(simulants, pop, number_of_simulants)
    else:
        simulants['age'] = initial_age
    simulants = create_sex_id_column(simulants, location_id, year_start)


    # TODO: Design and implement test that makes sure CEAM population looks
    # like population file pulled from GBD
    # TODO: Design and implement test that makes sure population has been
    # smoothed out-- JIRA TIC CE-213

    # assert an error to make sure data is dense (i.e. no missing data)
    assert simulants.isnull().values.any() == False, "there are nulls in the dataframe that generate_ceam_population just tried to output. check the function and its auxiliary functions (get_populations and assign_sex_id)"

    # assert an error if there are duplicate rows
    assert simulants.duplicated(['simulant_id']).sum(
    ) == 0, "there are duplicates in the dataframe that generate_ceam_population just tried to output. check the function and its auxiliary functions (get_populations and assign_sex_id)"

    return simulants


# 3. assign_cause_at_beginning_of_simulation

# TODO: Want to refactor the next few functions to assign prevalence at the beginning of the simulation in a better way
def get_cause_level_prevalence(states, location_id, year_start, draw_number):
    """
    Takes all of the sequela in 'states' and adds them up to get a total prevalence for the cause

    Parameters
    ----------
    states : dict
        dict with keys = name of cause, values = modelable entity id of cause

    location_id: int
        location_id for location of interest

    year_start: int
        year_start is the year in which the simulation will begin

    draw_number: int
        draw_number for this simulation run (specified in config file)

    Returns
    -------
    df with 1k draws where draw values are the prevalence of the cause of interest
    """
    prevalence_df = pd.DataFrame()
    prevalence_draws_dictionary = {}

    for key in states.keys():
        states[key] = states[key].query("year == {}".format(year_start))
        states[key] = states[key][['year', 'sex', 'age', 'draw_{}'.format(draw_number)]]
        prevalence_df = prevalence_df.append(states[key])

    cause_level_prevalence = prevalence_df.groupby(
        ['year', 'sex', 'age'], as_index=False).sum()

    return cause_level_prevalence, states


def determine_if_sim_has_cause(simulants_df, cause_level_prevalence, draw_number):
    """
    returns a dataframe with new column 'condition_envelope' that will indicate whether the simulant has the cause or is healthy (healthy is where condition_envelope = NaN at this point)

    Parameters
    ----------
    simulants_df: df
        dataframe of simulants that is made by generate_ceam_population

    cause_level_prevalence: df
        dataframe of 1k prevalence draws

    draw_number: int
        draw_number for this simulation run (specified in config file)

    Returns
    -------
    df with indication of whether or not simulant is healthy
    """
    merged = pd.merge(simulants_df, cause_level_prevalence, on=['age', 'sex'])
    probability_of_disease = merged['draw_{}'.format(draw_number)]
    probability_of_NOT_having_disease = 1 - probability_of_disease
    weights = np.array([probability_of_NOT_having_disease, probability_of_disease]).T

    results = simulants_df.copy()
    results['condition_envelope'] = choice('determine_if_sim_has_cause', simulants_df.simulant_id, [False, True], weights)

    return results


def get_sequela_proportions(prevalence_draws_dictionary, cause_level_prevalence, states, draw_number):
    """
    returns a dictionary with keys that are modelable entity ids and values are dataframes
    Parameters
    ----------
    prevalence_draws_dictionary: df
        dict of dataframes of simulants that contains where values are draws of sequela prevalence and keys are me_ids

    cause_level_prevalence: df
        dataframe of 1k prevalence draws

    draw_number: int
        draw_number for this simulation run (specified in config file)

    states : dict
        dict with keys = name of cause, values = modelable entity id of cause

    Returns
    -------
    A dictionary of dataframes where each dataframe contains proportion of cause prevalence made up by a specific sequela
    """
    sequela_proportions = {}

    # TODO: I do not think we want to be specifying draw_number here
    draw_number = config.getint('run_configuration', 'draw_number')

    for key in states.keys():
        sequela_proportions[key] = \
            pd.merge(prevalence_draws_dictionary[key], cause_level_prevalence, on=[
                'age', 'sex'], suffixes=('_single', '_total'))
        single = sequela_proportions[key][
            'draw_{}_single'.format(draw_number)]
        total = sequela_proportions[key][
            'draw_{}_total'.format(draw_number)]
        sequela_proportions[key]['scaled_prevalence'] = single / total

    return sequela_proportions


def determine_which_seq_diseased_sim_has(sequela_proportions, new_sim_file, states):
    """
    Parameters
    ----------
    sequela_proportions: dict
        a dictionary of dataframes where each dataframe contains proportion of cause prevalence made up by a specific sequela

    prevalence_draws_dictionary: df
        dict of dataframes of simulants that contains where values are draws of sequela prevalence and keys are me_ids

    new_sim_file: df
        dataframe of simulants

    states : dict
        dict with keys = name of cause, values = modelable entity id of cause

    Returns
    -------
    dataframe of simulants with new column condition_state that indicates if simulant which sequela simulant has or indicates that they are healthy (i.e. they do not have the disease)
    """
    sequela_proportions = [(key, Interpolation(data[['sex', 'age', 'scaled_prevalence']], ['sex'], ['age'])) for key, data in sequela_proportions.items()]
    sub_pop = new_sim_file.query('condition_envelope == 1')
    list_of_keys, list_of_weights = zip(*[(key,data(sub_pop)) for key, data in sequela_proportions])

    results = choice('determine_which_seq_diseased_sim_has', sub_pop.index, list_of_keys, np.array(list_of_weights).T)
    new_sim_file.loc[sub_pop.index, 'condition_state'] = results

    return new_sim_file

def assign_cause_at_beginning_of_simulation(simulants_df, location_id,
                                            year_start, states):
    """
    Function that assigns chronic ihd status to starting population of
    simulants

    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made by generate_ceam_population

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
    draw_number = config.getint('run_configuration', 'draw_number') 
    
    cause_level_prevalence, prevalence_draws_dictionary = get_cause_level_prevalence(states, location_id, year_start, draw_number) 

    # TODO: Should we be using groupby for these loops to ensure that we're
    # not looping over an age/sex combo that does not exist

    post_cause_assignment_population = determine_if_sim_has_cause(simulants_df, cause_level_prevalence, draw_number)    

    sequela_proportions = get_sequela_proportions(prevalence_draws_dictionary, cause_level_prevalence, states, draw_number)

    post_sequela_assignmnet_population = determine_which_seq_diseased_sim_has(sequela_proportions,  post_cause_assignment_population, states)

    post_sequela_assignmnet_population.condition_state = post_sequela_assignmnet_population.condition_state.fillna('healthy')

    # assert an error to make sure data is dense (i.e. no missing data)
    assert  post_sequela_assignmnet_population.isnull().values.any() == False, "there are nulls in the dataframe that assign_cause_at_beginning_of_simulation just tried to output. check that you've assigned the correct me_ids"

    # assert an error if there are duplicate rows
    assert  post_sequela_assignmnet_population.duplicated(['simulant_id']).sum(
    ) == 0, "there are duplicates in the dataframe that assign_cause_at_beginning_of_simulation just tried to output. check that you've assigned the correct me_ids"

    return post_sequela_assignmnet_population[['simulant_id', 'condition_state']]


# 4. get_cause_deleted_mortality_rate

def sum_up_csmrs_for_all_causes_in_microsim(df, list_of_me_ids, location_id,
                                            year_start, year_end):
    '''
    returns dataframe with columns for age, sex, year, and 1k draws
    the draws contain the sum of all the csmrs all of the causes in
    the current simulation.

    Parameters
    ----------
    df: df
        empty dataframe that will contain summed csmr_draws

    list_of_me_ids: list
        list of all of the me_ids in current simulation

    location_id: int
        to be passed into get_modelable_entity_draws

    year_start: int
        to be passed into get_modelable_entity_draws

    year_end: int
        to be passed into get_modelable_entity_draws

    Returns
    ----------
    df with columns year_id, sex_id, age, and draw_0 - draw_999
    '''
    for me_id in list_of_me_ids:
        csmr_draws = get_modelable_entity_draws(
            location_id, year_start, year_end, 15, me_id)
        df = df.append(csmr_draws)

    df = df.groupby(
        ['age', 'sex_id', 'year_id'], as_index=False).sum()

    return df


# 5. get_post_mi_heart_failure_proportion_draws


def get_post_mi_heart_failure_proportion_draws(location_id, year_start, year_end):
    # TODO: NEED TO WRITE TESTS TO MAKE SURE THAT POST_MI TRANSITIONS SCALE TO 1
    """
    Returns post-mi proportion draws for hf due to ihd

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int
        year_start is the year in which you want to start the simulation

    year_end : int
        year_end is the year in which you want to end the 

    Returns
    -------
    df with year_id, sex_id, age and 1k draws
    """

    # read in heart failure envelope. specify measure of interest
    hf_envelope = get_modelable_entity_draws(
        location_id, year_start, year_end, 6, 2412)

    # read in proportion of the cause of heart failure of interest
    proportion_draws = get_modelable_entity_draws(
        location_id, year_start, year_end, 18, 2414)

    # merge and then multiply envelope draws by proportion draws
    cause_of_hf = pd.merge(hf_envelope, proportion_draws, on=[
                           'age', 'year_id', 'sex_id'], suffixes=('_env', '_prop'))

    for i in range(0, 1000):
        # TODO: Manual calculation of the multiplication below gave a little bit different values. Should I be using np.multiply or somethig else to make sure python is handling these floats correctly?
        envelope = cause_of_hf['draw_{i}_env'.format(i=i)]
        proportion = cause_of_hf['draw_{i}_prop'.format(i=i)]
        # TODO: Make this block faster, have it calculate all probs for all draws in a single operation
        cause_of_hf['draw_{i}'.format(i=i)] = rate_to_probability(envelope * proportion)

    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert cause_of_hf.isnull().values.any() == False, "there are nulls in the dataframe that get_post_mi_heart_failure_proportion_draws just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert cause_of_hf.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_post_mi_heart_failure_proportion_draws just tried to output. check the cache to make sure that the data you're pulling is correct"

    # assert that none of the incidence rate values are greater than 1 (basically ensuring that the numerator and demoniator weren't flipped)
    draw_number = config.getint('run_configuration', 'draw_number')
    assert cause_of_hf['draw_{}'.format(draw_number)].all() <= 1, "something went wrong with the get_post_mi_heart_failure_proportion_draws calculation. incidence rate can't be GT 1. Check to see if the numerator/denominator were flipped"

    return cause_of_hf[keepcol]


# 6. get_relative_risks


def get_relative_risks(location_id, year_start, year_end, risk_id, cause_id, rr_type, gbd_round=1):
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

    rr_type: str
        can specify morbidity if you want RRs for incidence or mortality if you want RRs for mortality

    gbd_round: int
        # TODO: We'll need to update this when GBD 2016 is out. 
        generally set the gbd_round to 1 since this corresponds with GBD 2015
        there are some cases (e.g. pulling relative risks for unsafe sanitation)
        where the most recent (i.e. is best) model was updated in 2013 so we need
        to pass in the gbd_round variable to make sure that we're pulling the best
        model from that previous GBD round

    Returns
    -------
    df with columns year_id, sex_id, age, 1k relative risk draws
    """

    output_df = pd.DataFrame()

    rr = stata_wrapper('get_relative_risks.do', 'rel_risk_of_risk{r}_in_location{l}.csv'.format(r=risk_id,l=location_id), location_id, risk_id, cause_id)

    if rr_type == 'morbidity':
    elif rr_type == 'mortality':
    else:
        raise ValueError('rr type accepts one of two values, morbidity or mortality. you typed "{}" which is incorrect'.format(rr_type))

    rr = get_age_from_age_group_id(rr)

    # need to back calculate relative risk to earlier ages for risks that
    # don't start until a certain age.
    # TODO: Do we want to use an RR of 1 in the exposed groups? That's a pretty big assumption. It assumes that there is no risk of the risk factor on the exposure for those ages. If we don't have the data for the younger age groups, another alternative could be to backcast the relative risk of the youngest age group for which we do have data.
    output_df = rr.apply(lambda x: x.fillna(1), axis=0)

    keepcol = ['year_id', 'sex_id', 'age', 'parameter']
    keepcol.extend(('rr_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_relative_risks just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert that none of the rr values are less than 1
    draw_number = config.getint('run_configuration', 'draw_number')
    assert output_df['rr_{}'.format(draw_number)].all() >= 1, "something went wrong with get_relative_risks. RR cannot be LT 1. Check the data that you are pulling in and the function. Sometimes, the database does not have\
RR estimates for every age, so check to see that the function is correctly assigning relative risks to the other ages"

    return output_df[keepcol]


# 7. get_pafs

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

    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))
    output_df = pd.DataFrame(columns=keepcol)

    for sex_id in (1, 2):
        pafs = stata_wrapper('get_pafs.do', 'PAFs_for_{c}_in_{l}.csv'.format(c=cause_id, l=location_id), location_id, cause_id)

        # only want one risk at a time and only metric id 2 (percentages or pafs)
        pafs = pafs.query("rei_id == @risk_id and sex_id == @sex_id and metric_id == 2")

        if pafs.empty:
            continue

        pafs = get_age_from_age_group_id(pafs)

        pafs = set_age_year_index(pafs, 'early neonatal', 80, year_start, year_end)

        interp_data = interpolate_linearly_over_years_then_ages(pafs, 'draw')

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 105, year_start, year_end))

        # need to back calculate PAFS to earlier ages for risks that don't
        # start until a certain age
        output_df = output_df.apply(lambda x: x.fillna(0), axis=0)


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


    exposure = stata_wrapper('get_exposures.do', 'Exposure_of_risk{r}_in_location{l}.csv'.format(r=risk_id, l=location_id), location_id, risk_id)

    exposure = get_age_from_age_group_id(exposure)

    # TODO: Confirm that we want to be using cat1 here. Cat1 seems really high for risk_id=238 (handwashing without soap) for Kenya

    # TODO: Do we want to set the exposure to 0 for the younger ages for which we don't have data? It's an exceptionally strong assumption. We could use the exposure for the youngest age for which we do have data, or do something else, if we wanted to. --EM 12/12
    output_df = exposure.apply(lambda x: x.fillna(0), axis=0)

    keepcol = ['year_id', 'sex_id', 'age', 'parameter'] + ['draw_{i}'.format(i=i) for i in range(0, 1000)]

    # assert an error to make sure data is dense (i.e. no missing data)
    assert output_df.isnull().values.any() == False, "there are nulls in the dataframe that get_exposures just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id', 'parameter']).sum(
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


# 10. load_data_from_cache


memory = Memory(cachedir=get_cache_directory(), verbose=1)


@memory.cache
def _inner_cached_call(funct, *args, **kwargs):
    return funct(*args, **kwargs)


def load_data_from_cache(funct, col_name, *args, src_column=None, **kwargs):
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

    draw = config.getint('run_configuration', 'draw_number')

    if col_name:
        if src_column is not None:
            if isinstance(src_column, str):
                column_map = {src_column.format(draw=draw): col_name}
            else:
                column_map = {src.format(draw=draw):dest for src, dest in zip(src_column, col_name)}
        else:
            column_map = {'draw_{draw}'.format(draw=draw): col_name}

        # if 'parameter' is in columns, then keep it, else do not keep it (need parameter for the relative risk estimations)
        if 'parameter' in function_output.columns:
            keepcol = ['year_id', 'age', 'sex_id', 'parameter'] + list(column_map.keys())
        else:
            keepcol = ['year_id', 'age', 'sex_id'] + list(column_map.keys())

        function_output = function_output[keepcol]
        function_output = function_output.rename(columns=column_map)

        return normalize_for_simulation(function_output)
    return function_output


# 11. get_severity_splits


# 12. get_sbp_mean_sd

# TODO: write more unit tests for this function
def get_sbp_mean_sd(location_id, year_start, year_end):
    # TODO: Consider moving in the code from the blood pressure module
    # to here (i.e. interpolate from age 1 - 80, and fillna with the SBP values
    # we're using for under 25 yr olds)
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
    sbp_dir = os.path.join(get_cache_directory(), 'sbp')

    for sex_id in [1, 2]:
        draws = pd.DataFrame()
        for year_id in np.arange(year_start, year_end + 1, 5):
            path = auxiliary_file_path('Systolic Blood Pressure Distributions',
                                     location_id=location_id,
                                     year_id=year_id,
                                     sex_id=sex_id)
            one_year_file = pd.read_stata(path)
            one_year_file['year_id'] = year_id
            draws = draws.append(one_year_file)

        draws = get_age_from_age_group_id(draws)

        draws = set_age_year_index(draws, 'early neonatal', 80,
                                   year_start, year_end)

        interp_data = interpolate_linearly_over_years_then_ages(draws,
                                                                'exp_mean',
                                                                col_prefix2='exp_sd')

        interp_data['sex_id'] = sex_id

        #TODO: Need to rethink setting ages for this function. Since sbp estimates start for the age 25-29 group, it should start at age 25, not 27.5.
        # TODO: em python question -> best way to subset an index?
        # TODO: Make a list of columns before hand. will be faster

        # reset indexes to be columns and then assign sbp separately for young simulants
        interp_data.reset_index(level=['age', 'year_id'], inplace=True)
        young_simulants = interp_data.query("age < 27.5").copy()
        old_simulants = interp_data.query("age >= 27.5").copy()
        
        total_simulants = pd.DataFrame()        

        # FIXME: This process does produce a df that has null values for simulants under 27.5 years old for the exp_mean and exp_sd cols. Dont think this will affect anything but may be worth fixing
        for i in range(0, 1000):                     
            young_simulants['log_mean_{}'.format(i)] = np.log(112)
            young_simulants['log_sd_{}'.format(i)] = .001

            exp_mean = old_simulants['exp_mean_{}'.format(i)]
            exp_sd = old_simulants['exp_sd_{}'.format(i)]
            old_simulants['log_mean_{}'.format(i)] = np.log(exp_mean)
            old_simulants['log_sd_{}'.format(i)] = (exp_sd / exp_mean)
            
        total_simulants = total_simulants.append([young_simulants, old_simulants])

        total_simulants.set_index(['year_id', 'age'], inplace=True)

        output_df = output_df.append(
            extrapolate_ages(total_simulants, 105, year_start, year_end))

    # assert an error if there are duplicate rows
    assert output_df.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_sbp_mean_sd just tried to output. make sure what youre pulling from /share/epi/risk/paf/metab_sbp_interm/ is correct"

    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('log_mean_{i}'.format(i=i) for i in range(0, 1000)))
    keepcol.extend(('log_sd_{i}'.format(i=i) for i in range(0, 1000)))

    return output_df[keepcol].sort_values(by=['year_id', 'age', 'sex_id'])



def _bmi_ppf(parameters):
   return beta(a=parameters['a'], b=parameters['b'], scale=parameters['scale'], loc=parameters['loc']).ppf

@memory.cache
def get_bmi_distributions(location_id, year_start, year_end, draw):
    a = pd.DataFrame()
    b = pd.DataFrame()
    loc = pd.DataFrame()
    scale = pd.DataFrame()
    for sex_id in [1,2]:
        for year_id in np.arange(year_start, year_end + 1, 5):
            with open_auxiliary_file('Body Mass Index Distributions',
                                     parameter='bshape1',
                                     location_id=location_id,
                                     year_id=year_id,
                                     sex_id=sex_id) as f:
                a = a.append(pd.read_csv(f))
            with open_auxiliary_file('Body Mass Index Distributions',
                                     parameter='bshape2',
                                     location_id=location_id,
                                     year_id=year_id,
                                     sex_id=sex_id) as f:
                b = b.append(pd.read_csv(f))
            with open_auxiliary_file('Body Mass Index Distributions',
                                     parameter='mm',
                                     location_id=location_id,
                                     year_id=year_id,
                                     sex_id=sex_id) as f:
                loc = loc.append(pd.read_csv(f))
            with open_auxiliary_file('Body Mass Index Distributions',
                                     parameter='scale',
                                     location_id=location_id,
                                     year_id=year_id,
                                     sex_id=sex_id) as f:
                scale = scale.append(pd.read_csv(f))

    a = a.set_index(['age_group_id', 'sex_id', 'year_id'])
    b = b.set_index(['age_group_id', 'sex_id', 'year_id'])
    loc = loc.set_index(['age_group_id', 'sex_id', 'year_id'])
    scale = scale.set_index(['age_group_id', 'sex_id', 'year_id'])

    distributions = pd.DataFrame()
    distributions['a'] = a['draw_{}'.format(draw)]
    distributions['b'] = b['draw_{}'.format(draw)]
    distributions['loc'] = loc['draw_{}'.format(draw)]
    distributions['scale'] = scale['draw_{}'.format(draw)]

    distributions = distributions.reset_index()
    distributions = get_age_from_age_group_id(distributions)
    distributions['year'] = distributions.year_id
    distributions.loc[distributions.sex_id == 1, 'sex'] = 'Male'
    distributions.loc[distributions.sex_id == 2, 'sex'] = 'Female'

    return Interpolation(
            distributions[['age', 'year', 'sex', 'a', 'b', 'scale', 'loc']],
            categorical_parameters=('sex',),
            continuous_parameters=('age', 'year'),
            func=_bmi_ppf
            )


# 13 get_angina_proportions


def get_angina_proportions(year_start, year_end):
    '''Format the angina proportions so that we can use them in CEAM.
    This is messy. The proportions were produced by Catherine Johnson.
    The proportion differs by age, but not by sex, location, or time.
    This will likely change post GBD-2016.

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int
        year_start is the year in which you want to start the simulation


    Returns
    -------
    df with year_id, sex_id, age and 1k draws
    '''

    output_df = pd.DataFrame()

    for sex_id in [1, 2]:

        # TODO: Everett created csv below from a file that Catherine Johnson created
        # Catherine's original doc located here -- /snfs1/WORK/04_epi/01_database/02_data/cvd_ihd/04_models/02_misc_data/angina_prop_postMI.csv
        # Need to figure out a way to check to see if this file is ever updated
        with open_auxiliary_file('Angina Proportions') as f:
            ang = pd.read_csv(f)
        ang = ang.query("sex_id == {s}".format(s=sex_id))

        # TODO: After merging in pull request that allows for under 1 yr old estimation, change line below to read 'early neonatal' for age_start as opposed to 1
        indexed_ang = set_age_year_index(ang, 'early neonatal', 80, year_start, year_end)

        interp_data = interpolate_linearly_over_years_then_ages(indexed_ang, 'angina_prop')

        interp_data['sex_id'] = sex_id

        output_df = output_df.append(
            extrapolate_ages(interp_data, 105, year_start, year_end))

        # we don't have estimates under age 20, so I'm filling all ages under
        # 20 with the same proportion that we have for 20 year olds
        # TODO: Should check this assumption w/ Abie
    output_df = output_df.apply(lambda x: x.fillna(0.254902), axis=0)

    # little bit awkward below, but we're renaming the col name to have the draw number attached to it so that we can load it from the cache

    return output_df


# 14 get_disability_weight

def get_disability_weight(dis_weight_modelable_entity_id=None, healthstate_id=None):
    """Returns a dataframe with disability weight draws for a given healthstate id

    Parameters
    ----------
    dis_weight_modelable_entity_id : int

    Returns
    -------
    df with disability weight draws
    """

    if healthstate_id is None:
        healthstate_id = get_healthstate_id(dis_weight_modelable_entity_id)

    with open_auxiliary_file('Disability Weights') as f:
        dws_look_here_first = pd.read_csv(f)

    with open_auxiliary_file('Combined Disability Weights') as f:
        dws_look_here_second = pd.read_csv(f)

    if healthstate_id in dws_look_here_first.healthstate_id.tolist():
        df = dws_look_here_first.query("healthstate_id == @healthstate_id")
        df['modelable_entity_id'] = dis_weight_modelable_entity_id

    elif healthstate_id in dws_look_here_second.healthstate_id.tolist():
        df = dws_look_here_second.query("healthstate_id == @healthstate_id")
        df['modelable_entity_id'] = dis_weight_modelable_entity_id

    # TODO: Need to confirm with someone on central comp that all 'asymptomatic' sequala get this healthstate_id
    elif healthstate_id == 799:
        df = pd.DataFrame({'healthstate_id':[799], 'healthstate': ['asymptomatic'], 'modelable_entity_id':[dis_weight_modelable_entity_id], 'draw{}'.format(config.getint('run_configuration', 'draw_number')) : [0]})
    else:
        raise ValueError("""the modelable entity id {m} has a healthstate_id of {h}. it looks like there 
        are no draws for this healthstate_id in the csvs that get_healthstate_id_draws checked.
        look in this folder for the draws for healthstate_id{h}: /home/j/WORK/04_epi/03_outputs/01_code/02_dw/03_custom.
        if you can't find draws there, talk w/ central comp""".format(m=dis_weight_modelable_entity_id, h=healthstate_id)) 

    return df['draw{}'.format(config.getint('run_configuration', 'draw_number'))].iloc[0]

# 15. get_asympt_ihd_proportions
# TODO: Write a unit test for this function

def get_asympt_ihd_proportions(location_id, year_start, year_end):
    """
    Gets the proportion of post-mi simulants that will get asymptomatic ihd.
    Proportion that will get asymptomatic ihd is equal to 1 - proportion of 
    mi 1 month survivors that get angina + proportion of mi 1 month survivors
    that get heart failure

    Parameters
    ----------
    Feed in parameters required by get_post_mi_heart_failure_proportion_draws and get_angina_proportion_draws

    Returns
    -------
    df with post-mi asymptomatic ihd proportions
    """

    hf_prop_df = get_post_mi_heart_failure_proportion_draws(location_id, year_start, year_end)

    angina_prop_df = get_angina_proportions(year_start, year_end)

    asympt_prop_df = pd.merge(hf_prop_df, angina_prop_df, on=['age', 'year_id', 'sex_id'])
    
    # TODO: RAISE AN ERROR IF PROPORTIONS ARE GREATER THAN 1 FOR NOW. MAY WANT TO DELETE
    # ERROR IN THE FUTURE AND SCALE DOWN TO 1 INSTEAD
    angina_values = asympt_prop_df['angina_prop_{}'.format(config.getint('run_configuration', 'draw_number'))]

    for i in range(0, 1000):
        hf_values = asympt_prop_df['draw_{}'.format(i)]
        assert all(hf_values + angina_values) <= 1, "post mi proportions cannot be gt 1"      
        asympt_prop_df['asympt_prop_{}'.format(i)] = 1 - hf_values - angina_values
    
    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('asympt_prop_{i}'.format(i=i) for i in range(0, 1000)))

    return asympt_prop_df[keepcol] 


def get_age_specific_fertility_rates(location_id, year_start, year_end):
    #TODO: I'm loading this from disk because central comp doesn't have a good
    # tool for ingesting covariates from python and I don't feel like writing
    # any more stata. They say there should be something in a couple of weeks
    # and we should switch to it asap. -Alec 11/01/2016
    with open_auxiliary_file('Age-Specific Fertility Rates') as f:
        asfr = pd.read_csv(f)

    asfr = asfr.query('location_id == @location_id and year_id >= @year_start and year_id <= @year_end')
    asfr = get_age_from_age_group_id(asfr)

    return asfr


def get_etiology_pafs(location_id, year_start, year_end, risk_id, cause_id):
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
    # For some of the diarrhea etiologies, PAFs are negative. Wouldn't make sense for the simulation to use negative pafs (i.e. incidence * PAF returns a negative incidence if PAF is negative), so we'll clip to 0. Guessing that any other diseases that deal with etiologies in the future won't need to be treated this way. --EM 12/13
    # uses get pafs, but then scales the negative pafs to 0. the diarrhea team has some pafs that are negative because they wanted to include full uncertainty. this seems implausible in the real world, unless one is arguing that some pathogens have a protective effect

    eti_pafs = get_pafs(location_id, year_start, year_end, risk_id, cause_id)
 
    # now make the negative etiology paf draws 0
    draws = eti_pafs._get_numeric_data()
    draws[draws < 0] = 0

    return eti_pafs


def get_etiology_probability(etiology_name):
    """
    Gets the proportion of diarrhea cases that are associated with a specific etiology

    Parameters
    ----------
    etiology_name: str
        etiology_name is the name of the etiology of interest

    Returns
    -------
    """

    etiology_df = pd.DataFrame()

    # TODO: Ask Chris T. if this works for cholera and c diff, since they are modeled differently than the other etiologies
    # TODO: Will want to cache this data in the future instead of pulling it from Chris Troeger's J Temp file
    etiology_proportion_draws = pd.read_stata("/home/j/temp/ctroeger/Diarrhea/DALYs/Draws/diarrhea_{}_eti_draw_proportion.dta".format(etiology_name))

    etiology_proportion_draws = etiology_proportion_draws.query("location_id == {}".format(config.getint('simulation_parameters', 'location_id')))

    etiology_proportion_draws = get_age_from_age_group_id(etiology_proportion_draws)

    for sex in (1, 2):
        one_sex = etiology_proportion_draws.query("sex_id == @sex")
        # TODO: Figure out if we want to get info from the config in this script or elsewhere
        one_sex = set_age_year_index(one_sex, 'early neonatal', 3, config.getint('simulation_parameters', 'year_start') , config.getint('simulation_parameters', 'year_end'))

        etiology_df = etiology_df.append(one_sex)

    etiology_df.reset_index(level=['age', 'year_id'], inplace=True)

    keepcol = ['year_id', 'sex_id', 'age', 'draw_{i}'.format(i=config.getint('run_configuration', 'draw_number'))]

    return etiology_df[keepcol]


def get_etiology_and_severity_specific_incidence(location_id, year_start, year_end, eti_risk_id, cause_id, me_id):
    """
    Gets the paf of diarrhea cases that are associated with a specific etiology

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    eti_risk_id: int, risk id
        eti_risk_id takes same rei id values as are used for GBD

    cause_id: int, cause id
        cause_id takes same cause_id values as are used for GBD

    me_id: int, modelable_entity_id
        me_id takes modelable entity id of a cause

    Returns
    -------
    A dataframe of etiology specific incidence draws.
        Column are age, sex_id, year_id, and {etiology_name}_incidence_{draw} (1k draws)
    """

    # TODO: Figure out what we want to do regarding caching data. If we are using only one draw number, should we 
    # cache the data? If we cache the data, will we actually pull a different value for each draw
    draw_number = config.getint('run_configuration', 'draw_number')
    
    diarrhea_envelope_incidence = get_modelable_entity_draws(location_id, year_start, year_end,
                                                           measure=6, me_id=me_id) # measure=incidence, me_id=diarrhea envelope TODO: Make me_id an argument to be passed into the fucntion (i.e. make this function more flexible than just diarrhea)

    etiology_paf = get_etiology_pafs(location_id, year_start, year_end, eti_risk_id, cause_id)
    
    # TODO: Figure out if the interpolation should happen before the merge or in the simulation
    # merge interpolated pafs and interpolated incidence
    etiology_specific_incidence= pd.merge(diarrhea_envelope_incidence, etiology_paf, on=['age', 'year_id', 'sex_id'],
                                          suffixes=('_envelope', '_pafs'))
    
    # TODO: Do all 1,000 draws at once or just return 1 draw?
    paf = etiology_specific_incidence['draw_{}_pafs'.format(draw_number)]
    env_inc = etiology_specific_incidence['draw_{}_envelope'.format(draw_number)]
    etiology_specific_incidence['draw_{}'.format(draw_number)] = np.multiply(paf, env_inc)

    keepcol = ['year_id', 'sex_id', 'age', 'draw_{}'.format(draw_number)]

    return etiology_specific_incidence[keepcol]


def get_etiology_and_severity_specific_prevalence(location_id, year_start, year_end, eti_risk_id, cause_id, me_id):
    """
    Gets draws of prevalence of diarrhea due to a specific specific etiology

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

    me_id: int, modelable_entity_id
        me_id takes modelable entity id of a cause

    Returns
    -------
    A dataframe of etiology specific prevalence draws. 
        Column are age, sex_id, year_id, and {etiology_name}_incidence_{draw} (1k draws)
    """

    draw_number = config.getint('run_configuration', 'draw_number')   
 
    diarrhea_envelope_prevalence = get_modelable_entity_draws(location_id, year_start, year_end,
                                                           measure=5, me_id=me_id) # measure=prevalence, me_id=diarrhea envelope
    
    etiology_paf = get_pafs(location_id, year_start, year_end, eti_risk_id, cause_id)

    etiology_specific_prevalence= pd.merge(etiology_paf, diarrhea_envelope_prevalence, on=['age', 'year_id', 'sex_id'], 
                                          suffixes=('_envelope', '_pafs'))

    paf = etiology_specific_prevalence['draw_{}_pafs'.format(draw_number)]
    env_prev = etiology_specific_prevalence['draw_{}_envelope'.format(draw_number)]
    etiology_specific_prevalence['draw_{}'.format(draw_number)] = np.multiply(paf, env_prev)

    keepcol = ['year_id', 'sex_id', 'age', 'draw_{}'.format(draw_number)]

    return etiology_specific_prevalence[keepcol]


# TODO: Figure out if we need to do anything to the remission rates. We have remission for all diarrhea.
# Do we need to split out remission to get remission from the different severity states?7
def get_diarrhea_severity_split_excess_mortality(excess_mortality_dataframe, severity_split):
    if severity_split == 'severe':
        # FIXME: Need to use severity split draws. Manually setting proportions for now
        severe_diarrhea_proportion = .14
        excess_mortality_dataframe['rate'] = excess_mortality_dataframe['rate'] / severe_diarrhea_proportion
    elif severity_split in ['mild', 'moderate']:
        # set the excess mortality rate to 0
        excess_mortality_dataframe['rate'] = 0
    else:
        raise ValueError("you supplied an invalid value for severity split argument. you wrote '{}'. acceptable severity splits are mild, moderate, or severe".format(severity_split))
    return excess_mortality_dataframe


# End.
