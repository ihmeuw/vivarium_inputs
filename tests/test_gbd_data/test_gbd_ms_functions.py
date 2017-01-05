# ~/ceam/ceam_tests/test_gbd_data/test_gbd_ms_functions.py

import pytest
import numpy as np
import ceam.gbd_data.gbd_ms_functions
from ceam.gbd_data.gbd_ms_functions import get_cause_level_prevalence
from ceam.gbd_data.gbd_ms_functions import get_relative_risks
from ceam.gbd_data.gbd_ms_functions import get_pafs
from ceam.gbd_data.gbd_ms_functions import get_exposures
from ceam.gbd_data.gbd_ms_functions import get_angina_proportions

def test_get_sbp_mean_sd_Kenya_2000():
    # set the parameters
    location_id = 180 # Kenya
    year_start = 2000
    year_end = 2000

    # load the sbp data
    df = ceam.gbd_data.gbd_ms_functions.get_sbp_mean_sd(location_id, year_start, year_end)

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


# em's unit tests
# 1. get_modelable_entity_draws
    # no unit test needed for this function. will produce graphs instead


# 2. assign_cause_at_the_beginning_of_the_simulation
    # need to prove that total cause envelope (e.g. ihd) is correct
	# need to prove that each sequelae is correctly assigned
            # these should both be done graphically

# 2a. get_cause_level_prevalence
def test_get_cause_level_prevalence():
    # pass in a states dict with only two sequela and make sure for one age/sex/year combo that the value in cause_level_prevalence is equal to the sum of the two seq prevalences
    dict_of_disease_states = {'severe_heart_failure' : 1823, 'moderate_heart_failure' : 1822}    
    
    # pick a random draw to test
    draw_number = 19 
    cause_level, seq_level_dict = get_cause_level_prevalence(dict_of_disease_states, 180, 1990, draw_number)

    # pick a random age and sex to test
    sex = 1
    age = 42

    # get a prevaelnce estimate for the random age and sex that we want to test
    moderate_heart_failure = seq_level_dict['moderate_heart_failure'].query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    seq_prevalence_1 = moderate_heart_failure['draw_{}'.format(draw_number)].values[0]
    severe_heart_failure = seq_level_dict['severe_heart_failure'].query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    seq_prevalence_2 = severe_heart_failure['draw_{}'.format(draw_number)].values[0]
    
    # add up the prevalences of the 2 sequela to see if we get cause-level prevalence
    cause_level = cause_level.query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    cause_prev = cause_level['draw_{}'.format(draw_number)].values[0]    
    
    assert cause_prev == seq_prevalence_1 + seq_prevalence_2, 'get_cause_level_prevalence error. seq prevs need to add up to cause prev'


# 3. get_relative_risks
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


# 4. get_pafs
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
    assert sd == .001 "sbp standard deviation should be .001 for simulants under age 27.5"


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

    assert np.allclose(props.get_value(7.5, 'angina_prop'), props.get_value(22.5, 'angina_prop'), "get_angina_proportions needs to assign values for people younger than age group 9 to get the same value as people in age group 9"

    assert np.allclose(props.get_value(82.5, 'angina_prop') == 0.128526646, "get_angina_proportions needs to return values that match input file" 


# End.
