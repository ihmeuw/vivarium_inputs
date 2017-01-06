# ~/ceam/ceam_tests/test_gbd_data/test_gbd_ms_functions.py

import pytest
import numpy as np
import pandas as pd
from ceam_inputs.gbd_ms_functions import get_sbp_mean_sd
from ceam_inputs.gbd_ms_functions import get_relative_risks
from ceam_inputs.gbd_ms_functions import get_pafs
from ceam_inputs.gbd_ms_functions import get_exposures
from ceam_inputs.gbd_ms_functions import get_angina_proportions
from ceam_inputs.gbd_ms_functions import get_disability_weight
from ceam_inputs import generate_ceam_population
from ceam_inputs.gbd_ms_functions import get_cause_level_prevalence
from ceam_inputs import get_prevalence
from ceam_inputs.gbd_ms_functions import determine_if_sim_has_cause
from ceam_inputs.gbd_ms_functions import get_sequela_proportions
from ceam_inputs.gbd_ms_functions import determine_which_seq_diseased_sim_has

def test_get_sbp_mean_sd_Kenya_2000():
    # set the parameters
    location_id = 180 # Kenya
    year_start = 2000
    year_end = 2000

    # load the sbp data
    df = get_sbp_mean_sd(location_id, year_start, year_end)

    df = df[['year_id', 'sex_id', 'age', 'log_mean_0']]

    # reshape it so it is easy to access
    df = df.groupby(['year_id', 'sex_id', 'age']).first()

    # check if the value for 25 year old males matches the csv
    assert np.allclose(df.loc[(2000, 1, 27.5), 'log_mean_0'], np.log(118.948299)), 'should match data loaded by @aflaxman on 8/4/2016. test changed by @emumford on 9/23 to account for change in gbd_ms_functions'


@pytest.mark.xfail
def test_get_sbp_mean_sd_2001():
    # load the sbp data
    df = ceam.gbd_data.gbd_ms_functions.get_sbp_mean_sd(80, 2001, 2001)
    # would be nice if this worked

# generate_ceam_population
def test_generate_ceam_population():
    pop = generate_ceam_population(500000)

    num_7_and_half_yr_old_males = pop.query("age == 7.5 and sex_id == 1").copy()

    num_7_and_half_yr_old_males['count'] = 1

    val = num_7_and_half_yr_old_males.groupby('age')[['count']].sum()

    val = val.get_value(7.5, 'count')

    val = val / 500000

    assert np.allclose(val, 0.0823207075530383, .01), "there should be about 8.23% 7.5 year old males in Kenya in 1990, based on data uploaded by em 1/5/2017"

# get_cause_level_prevalence
def test_get_cause_level_prevalence():
    # pass in a states dict with only two sequela and make sure for one age/sex/year combo that the value in cause_level_prevalence is equal to the sum of the two seq prevalences
    dict_of_disease_states = {'severe_heart_failure' : get_prevalence(1823), 'moderate_heart_failure' : get_prevalence(1822)}    
     
    cause_level, seq_level_dict = get_cause_level_prevalence(dict_of_disease_states, 180, 1990)

    # pick a random age and sex to test
    sex = "Male"
    age = 42.5

    # get a prevalence estimate for the random age and sex that we want to test
    moderate_heart_failure = seq_level_dict['moderate_heart_failure'].query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_1 = moderate_heart_failure['prevalence'].values[0]
    severe_heart_failure = seq_level_dict['severe_heart_failure'].query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_2 = severe_heart_failure['prevalence'].values[0]
    
    # add up the prevalences of the 2 sequela to see if we get cause-level prevalence
    cause_level = cause_level.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    cause_prev = cause_level['prevalence'].values[0]    
    
    assert cause_prev == seq_prevalence_1 + seq_prevalence_2, 'get_cause_level_prevalence error. seq prevs need to add up to cause prev'
    assert np.allclose(cause_prev, 0.00010368 + 0.00022846), 'get_cause_level prevalence should match data from database as of 1/5/2017' 

# determine_if_sim_has_cause
def test_determine_if_sim_has_cause():
    prevalence_df = pd.DataFrame({"age": [0, 5, 10, 15], "sex": ['Male']*4 , "prevalence": [.25, .5, .75, 1]})

    simulants_df = pd.DataFrame({'simulant_id': range(0, 500000), 'sex': ['Male']*500000, 'age': [0, 5, 10, 15]*125000})

    results = determine_if_sim_has_cause(simulants_df, prevalence_df)

    grouped_results = results.groupby('age')[['condition_envelope']].sum()

    assert np.allclose(grouped_results.get_value(0, 'condition_envelope')/125000, .25, .01), "determine if sim has cause needs to appropriately assign causes based on prevalence"
    
    assert np.allclose(grouped_results.get_value(5, 'condition_envelope')/125000, .5, .01), "determine if sim has cause needs to appropriately assign causes based on prevalence"

    assert np.allclose(grouped_results.get_value(10, 'condition_envelope')/125000, .75, .01), "determine if sim has cause needs to appropriately assign causes based on prevalence"

    assert np.allclose(grouped_results.get_value(15, 'condition_envelope')/125000, 1), "determine if sim has cause needs to appropriately assign causes based on prevalence"  


# get_sequela_proportions
def test_get_sequela_proportions():
    cause_level_prevalence = pd.DataFrame({"age": [0, 5, 10, 15], "sex": ['Male']*4 , "prevalence": [.25, .5, .75, 1], "year": 1990})
    
    seq_1_prevalence_df = cause_level_prevalence.copy()
    seq_2_prevalence_df = cause_level_prevalence.copy()
    
    seq_1_prevalence_df.prevalence = seq_1_prevalence_df['prevalence'] * .75
    seq_2_prevalence_df.prevalence = seq_2_prevalence_df['prevalence'] * .25
    
    states = dict({'sequela 1': seq_1_prevalence_df, 'sequela 2': seq_2_prevalence_df})
    
    df = get_sequela_proportions(cause_level_prevalence, states)
    
    assert list(df['sequela 1'].scaled_prevalence.values) == [.75]*4, "get_sequela_proportions"
    assert list(df['sequela 2'].scaled_prevalence.values) == [.25]*4, "get_sequela_proportions"


# determine_which_seq_diseased_sim_has
def test_determine_which_seq_diseased_sim_has():
    simulants_df = pd.DataFrame({'age': [0]*200000, 'sex': ['Male']*200000, 'simulant_id': range(0,200000), 'condition_envelope': [False, True]*100000})
    
    df1 = pd.DataFrame({'age': [0, 10, 0, 10], 'sex': ['Male']*2 + ['Female']*2, 'scaled_prevalence': [.75, 1, .75, 1] })
    
    df2 = pd.DataFrame({'age': [0, 10, 0, 10], 'sex': ['Male']*2 + ['Female']*2, 'scaled_prevalence': [.25, 0, .25, 0] })
    
    sequela_proportion_dict = dict({'sequela 1': df1, 'sequela 2': df2})
    
    results = determine_which_seq_diseased_sim_has(sequela_proportion_dict, simulants_df)
    
    results['count'] = 1
    
    seq1 = results.query("condition_state == 'sequela 1'")
    seq2 = results.query("condition_state == 'sequela 2'")
    
    val1 = seq1.groupby('age')[['count']].sum()
    val1 = val1.get_value(0, 'count')
    val1 = val1 / 100000
    
    val2 = seq2.groupby('age')[['count']].sum()
    val2 = val2.get_value(0, 'count')
    val2 = val2 / 100000
    
    assert np.allclose(val1, .75, .1), "determine which seq diseased sim has needs to assign sequelas according to sequela prevalence"
    assert np.allclose(val2, .25, .1), "determine which seq diseased sim has needs to assign sequelas according to sequela prevalence" 


# get_relative_risks
def test_get_relative_risks():
    df = get_relative_risks(180, 1990, 1990, 107, 493)

    # pick a random draw to test
    draw_number = 19
    
    # pick a random age under 25 and sex to test
    sex = 2
    age = 19

    # assert that relative risks are 1 for people under age 25 for high sbp
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))
    rr = df_filter['rr_{}'.format(draw_number)].values[0]

    assert rr == 1.0, 'get_relative_risks should return rr=1 for younger ages for the risks which dont estimate relative risk for all ages'


# get_pafs
def test_get_pafs():
    df = get_pafs(180, 1990, 1990, 107, 493) 

    # pick a random draw to test
    draw_number = 19

    # pick a random age under 25 and sex to test
    sex = 1
    age = 19

    # assert that pafs are 0 for people under age 25 for high sbp
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    pafs = df_filter['draw_{}'.format(draw_number)].values[0]

    assert pafs == 0, 'get_pafs should return paf=0 for the ages for which we do not have GBD estimates'


# get_exposures
def test_get_exposures():
    df = get_exposures(180, 1990, 1990, 107)

    # pick a random draw to test
    draw_number = 999

    # pick a random age under 25 and sex to test
    sex = 2
    age = 19

    # assert that exposures are 0 for people under age 25 for high sbp
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    exposure = df_filter['draw_{}'.format(draw_number)].values[0]

    assert exposure == 0, 'get_exposure should return exposure=0 for the ages for which we do not have GBD estimates'


def test_get_sbp_mean_sd():
    df = get_sbp_mean_sd(163, 1990, 1990)

    # pick a random draw to test
    draw_number = 114

    # pick a random age under 25 and sex to test
    sex = 1
    age = 9

    # assert that sbp = log(112) and sd = .001 for people under age 25
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))
    sbp = df_filter['log_mean_{}'.format(draw_number)].values[0]
    sd = df_filter['log_sd_{}'.format(draw_number)].values[0]

    assert sbp == np.log(112), "sbp for simulants under age 27.5 should equal log of 112"
    assert sd == .001, "sbp standard deviation should be .001 for simulants under age 27.5"


def test_sum_up_csmrs_for_all_causes_in_microsim():
    csmr1 = get_modelable_entity_draws(
            180, 1990, 1990, 15, 3233)

    csmr2 = get_modelable_entity_draws(
            180, 1990, 1990, 15, 1814)

    sex = 2
    age = 72
    draw_number = 77

    csmr1_filter = csmr1.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    csmr2_filter = csmr2.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    csmr1_val = csmr1_filter['draw_{}'.format(draw_number)].values[0]

    csmr2_val = csmr2_filter['draw_{}'.format(draw_number)].values[0]

    df = sum_up_csmrs_for_all_causes_in_microsim([1814, 3233], 180, 1990, 1990)

    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    df_val = df_filter['draw_{}'.format(draw_number)].values[0]

    assert df_val == csmr1_val + csmr2_val, "sum_up_csmrs_for_all_causes_in_microsim did not correctly sum up csmrs"


def test_get_cause_deleted_mortality_rate():
    all_cause_mr = get_all_cause_mortality_rate(180, 1990, 1990)

    sex = 1 
    age = 67
    draw_number = 221

    all_cause_filter = all_cause_mr.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    cause_csmr = sum_up_csmrs_for_all_causes_in_microsim([3233], 180, 1990, 1990)

    csmr_filter = cause_csmr.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    all_cause_val = all_cause_filter['all_cause_mortality_rate_{}'.format(draw_number)].values[0]
    
    cause_val = csmr_filter['draw_{}'.format(draw_number)].values[0]

    cause_deleted = get_cause_deleted_mortality_rate(180, 1990, 1990, [3233])

    cause_deleted_filter = cause_deleted.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    cause_deleted_val = cause_deleted_filter['cause_deleted_mortality_rate_{}'.format(draw_number)].values[0]

    assert cause_deleted_val == all_cause_val - cause_val, "cause deleted mortality rate was incorrectly calculated"


def test_get_angina_proportions():
    
    angina_props = get_angina_proportions()

    props.set_index('age', inplace=True)

    props = props.query('sex_id == 1')

    assert np.allclose(props.get_value(7.5, 'angina_prop'), props.get_value(22.5, 'angina_prop')), "get_angina_proportions needs to assign values for people younger than age group 9 to get the same value as people in age group 9"

    assert np.allclose(props.get_value(82.5, 'angina_prop'), 0.128526646), "get_angina_proportions needs to return values that match input file" 


def test_get_disability_weight():
    # me_id 2608 = mild diarrhea
    assert np.allclose(get_disability_weight(dis_weight_modelable_entity_id=2608), 0.0983228), "get_disability_weight should return the correct disability weight from the flat files prepared by central comp"


# End.
