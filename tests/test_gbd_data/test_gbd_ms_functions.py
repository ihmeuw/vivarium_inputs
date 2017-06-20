from unittest.mock import patch
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ceam import config
from ceam.framework.util import rate_to_probability
from ceam_tests.util import build_table

from ceam_inputs import get_cause_specific_mortality, get_excess_mortality, causes, risk_factors
from ceam_inputs.gbd_ms_functions import (get_sbp_mean_sd, get_relative_risks, get_pafs, get_exposures,
                                          get_angina_proportions, get_disability_weight, generate_ceam_population,
                                          get_cause_level_prevalence, determine_if_sim_has_cause,
                                          get_sequela_proportions, determine_which_seq_diseased_sim_has,
                                          get_post_mi_heart_failure_proportion_draws, get_modelable_entity_draws,
                                          sum_up_csmrs_for_all_causes_in_microsim, get_cause_deleted_mortality_rate,
                                          assign_subregions, get_etiology_specific_incidence,
                                          get_etiology_specific_prevalence, get_asympt_ihd_proportions,
                                          normalize_for_simulation, get_severe_diarrhea_excess_mortality,
                                          get_rota_vaccine_coverage, get_mediation_factors)

KENYA = 180


def test_generate_ceam_population():
    np.random.seed(1430)
    pop = generate_ceam_population(location_id=KENYA,
                                   time=datetime(1990, 1, 1),
                                   number_of_simulants=1000000,
                                   gbd_round_id=3,
                                   pop_age_start=0,
                                   pop_age_end=110)

    num_7_and_half_yr_old_males = pop.query("age == 7.5 and sex_id == 1").copy()
    num_7_and_half_yr_old_males['count'] = 1
    val = num_7_and_half_yr_old_males.groupby('age')[['count']].sum()
    val = val.get_value(7.5, 'count')
    val = val / 1000000

    assert np.isclose(val, 0.0823207075530383, atol=.01), ("there should be about 8.23% 7.5 year old males in "
                                                           "Kenya in 1990, based on data uploaded by em 1/5/2017")


def test_get_cause_level_prevalence():
    # pass in a states dict with only two sequela and make sure for one age/sex/year combo
    # that the value in cause_level_prevalence is equal to the sum of the two seq prevalences
    prev_df1 = build_table(0.03).rename(columns={'rate': 'prevalence'})[['year', 'age', 'prevalence', 'sex']]
    prev_df2 = build_table(0.02).rename(columns={'rate': 'prevalence'})[['year', 'age', 'prevalence', 'sex']]

    dict_of_disease_states = {'severe_heart_failure': prev_df1, 'moderate_heart_failure': prev_df2}
    cause_level, seq_level_dict = get_cause_level_prevalence(dict_of_disease_states, year_start=2005)

    # pick a random age and sex to test
    sex = "Male"
    age = 42

    # get a prevalence estimate for the random age and sex that we want to test
    moderate_heart_failure = seq_level_dict['moderate_heart_failure'].query(
        "age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_1 = moderate_heart_failure['prevalence'].values[0]
    severe_heart_failure = seq_level_dict['severe_heart_failure'].query(
        "age == {a} and sex == '{s}'".format(a=age, s=sex))
    seq_prevalence_2 = severe_heart_failure['prevalence'].values[0]

    # add up the prevalences of the 2 sequela to see if we get cause-level prevalence
    cause_level = cause_level.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    cause_prev = cause_level['prevalence'].values[0]

    assert np.isclose(cause_prev, seq_prevalence_1 + seq_prevalence_2), ('get_cause_level_prevalence error. '
                                                                         'seq prevs need to add up to cause prev')
    assert np.allclose(cause_prev, .05), 'get_cause_level prevalence should match data from database as of 1/5/2017'


# determine_if_sim_has_cause
def test_determine_if_sim_has_cause():
    prevalence_df = pd.DataFrame({"age": [0, 5, 10, 15],
                                  "sex": ['Male']*4,
                                  "prevalence": [.25, .5, .75, 1],
                                  "year": [1990]*4})
    simulants_df = pd.DataFrame({'simulant_id': range(0, 500000),
                                 'sex': ['Male']*500000,
                                 'age': [0, 5, 10, 15]*125000})
    results = determine_if_sim_has_cause(simulants_df, prevalence_df)
    grouped_results = results.groupby('age')[['condition_envelope']].sum()

    err_msg = "determine if sim has cause needs to appropriately assign causes based on prevalence"
    assert np.allclose(grouped_results.get_value(0, 'condition_envelope')/125000, .25, .01), err_msg
    assert np.allclose(grouped_results.get_value(5, 'condition_envelope')/125000, .5, .01), err_msg
    assert np.allclose(grouped_results.get_value(10, 'condition_envelope')/125000, .75, .01), err_msg
    assert np.allclose(grouped_results.get_value(15, 'condition_envelope')/125000, 1), err_msg


# get_sequela_proportions
def test_get_sequela_proportions():
    cause_level_prevalence = pd.DataFrame({"age": [0, 5, 10, 15],
                                           "sex": ['Male']*4,
                                           "prevalence": [.25, .5, .75, 1],
                                           "year": 1990})

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
    simulants_df = pd.DataFrame({'age': [0]*200000,
                                 'sex': ['Male']*200000,
                                 'simulant_id': range(200000),
                                 'condition_envelope': [False, True]*100000})

    df1 = pd.DataFrame({'age': [0, 10, 0, 10],
                        'sex': ['Male']*2 + ['Female']*2,
                        'scaled_prevalence': [.75, 1, .75, 1]})
    df2 = pd.DataFrame({'age': [0, 10, 0, 10],
                        'sex': ['Male']*2 + ['Female']*2,
                        'scaled_prevalence': [.25, 0, .25, 0]})
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

    err_msg = "determine which seq diseased sim has needs to assign sequelae according to sequela prevalence"
    assert np.allclose(val1, .75, .1), err_msg
    assert np.allclose(val2, .25, .1), err_msg


def test_get_post_mi_heart_failure_proportion_draws():
    df = get_post_mi_heart_failure_proportion_draws(location_id=KENYA,
                                                    year_start=1990,
                                                    year_end=2015,
                                                    draw_number=0)
    err_msg = ("get_post_mi_heart_failure proportion draws needs to return the correct "
               "proportion of simulants that will have heart failure after suffering an mi")
    assert np.isclose(df.get_value(199, 'draw_0'), rate_to_probability(np.multiply(0.16197485, 0.01165705))), err_msg
    assert np.isclose(df.get_value(116, 'draw_10'), rate_to_probability(np.multiply(0.00839225, 0.2061004))), err_msg


def test_get_relative_risks():
    df = get_relative_risks(location_id=KENYA,
                            year_start=1990,
                            year_end=1990,
                            risk_id=risk_factors.high_systolic_blood_pressure.gbd_risk,
                            cause_id=causes.heart_attack.gbd_cause,
                            gbd_round_id=3,
                            draw_number=0,
                            rr_type='morbidity')
    draw_number = 19  # Arbitrary selection

    df_filter1 = df.query("age == 7.5 and sex_id == 2")
    df_filter1.set_index('age', inplace=True)
    rr1 = df_filter1.get_value(7.5, 'rr_{}'.format(draw_number))
    df_filter2 = df.query("age == 82.5 and sex_id == 2")
    df_filter2.set_index('age', inplace=True)
    rr2 = df_filter2.get_value(82.5, 'rr_{}'.format(draw_number))

    assert np.allclose(rr1, 1.0), ("get_relative_risks should return rr=1 for younger ages "
                                   "for the risks which don't estimate relative risk for all ages")
    assert np.allclose(rr2, 1.3506), 'get_relative risks should return rrs that match what is pulled from the database'


def test_get_pafs():
    df = get_pafs(location_id=KENYA,
                  year_start=1990,
                  year_end=1990,
                  risk_id=risk_factors.high_systolic_blood_pressure.gbd_risk,
                  cause_id=causes.heart_attack.gbd_cause,
                  gbd_round_id=3,
                  draw_number=0,
                  paf_type='morbidity')
    draw_number = 19
    # assert that pafs are 0 for people under age 25 for high sbp
    df_filter1 = df.query("age == 7.5 and sex_id == 2")
    df_filter1.set_index('age', inplace=True)
    paf1 = df_filter1.get_value(7.5, 'draw_{}'.format(draw_number))
    df_filter2 = df.query("age == 82.5 and sex_id == 2")
    df_filter2.set_index('age', inplace=True)
    paf2 = df_filter2.get_value(82.5, 'draw_{}'.format(draw_number))

    assert paf1 == 0, 'get_pafs should return paf=0 for the ages for which we do not have GBD estimates'
    assert np.isclose(paf2, 0.64621693), 'get_pafs should return pafs that match what is pulled from the database'


def test_get_exposures():
    df = get_exposures(location_id=KENYA,
                       year_start=1990,
                       year_end=1990,
                       risk_id=risk_factors.smoking_prevalence_approach.gbd_risk,
                       gbd_round_id=3)

    # assert that exposures are 0 for people under age 25 for high sbp
    df_filter1 = df.query("age == 7.5 and sex_id == 2 and parameter == 'cat1'")
    df_filter1.set_index('age', inplace=True)
    exposure1 = df_filter1.get_value(7.5, 'draw_0')
    df_filter2 = df.query("age == 82.5 and sex_id == 2 and parameter == 'cat1'")
    df_filter2.set_index('age', inplace=True)
    exposure2 = df_filter2.get_value(82.5, 'draw_0')

    assert exposure1 == 0, 'get_exposure should return exposure=0 for the ages for which we do not have GBD estimates'
    assert np.isclose(exposure2, 0.03512375), ('get_exposures should return exposures '
                                               'that match what is pulled from the database')


def test_get_sbp_mean_sd():
    df = get_sbp_mean_sd(location_id=KENYA, year_start=1990, year_end=1990)
    draw_number = 114
    sex = 1
    age = 7.5

    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))
    sbp = df_filter['log_mean_{}'.format(draw_number)].values[0]
    sd = df_filter['log_sd_{}'.format(draw_number)].values[0]

    assert sbp == np.log(112), "sbp for simulants under age 27.5 should equal log of 112"
    assert sd == .001, "sbp standard deviation should be .001 for simulants under age 27.5"


@pytest.mark.xfail(strict=True)
def test_get_sbp_mean_sd_kenya_2000():
    df = get_sbp_mean_sd(location_id=KENYA, year_start=2000, year_end=2000)
    df = df[['year_id', 'sex_id', 'age', 'log_mean_0']]
    df = df.groupby(['year_id', 'sex_id', 'age']).first()

    err_msg = ('should match data loaded by @aflaxman on 8/4/2016. test changed by '
               '@emumford on 9/23 to account for change in gbd_ms_functions')
    assert np.allclose(df.loc[(2000, 1, 27.5), 'log_mean_0'], np.log(118.948299)), err_msg


def test_sum_up_csmrs_for_all_causes_in_microsim():
    csmr1 = get_cause_specific_mortality(causes.heart_attack.gbd_cause)
    csmr2 = get_cause_specific_mortality(causes.ischemic_stroke.gbd_cause)

    sex = 'Female'
    age = 72.5

    csmr1_filter = csmr1.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    csmr2_filter = csmr2.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    csmr1_val = csmr1_filter['rate'].values[0]
    csmr2_val = csmr2_filter['rate'].values[0]

    df = sum_up_csmrs_for_all_causes_in_microsim([csmr1_filter, csmr2_filter])
    df_filter = df.query("age == {a} and sex == '{s}'".format(a=age, s=sex))
    df_val = df_filter['rate'].values[0]

    assert df_val == csmr1_val + csmr2_val, "sum_up_csmrs_for_all_causes_in_microsim did not correctly sum up csmrs"


def test_get_cause_deleted_mortality_rate():
    age = 67.5

    location_id = config.simulation_parameters.location_id
    year_start = config.simulation_parameters.year_start
    year_end = config.simulation_parameters.year_end
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number

    # FIXME: add all cause mortality to the gbd_mapping suite
    all_cause_mr = get_cause_specific_mortality(causes.all_causes.gbd_cause)
    all_cause_mr = all_cause_mr[['age', 'sex', 'year', 'rate']]
    all_cause_mr.columns = ['age', 'sex', 'year', 'all_cause_mortality_rate']

    all_cause_filter = all_cause_mr.query("age == {a} and sex == 'Male' and year == {y}".format(a=age, y=year_start))
    all_cause_val = all_cause_filter['all_cause_mortality_rate'].values[0]

    csmr493 = get_cause_specific_mortality(causes.ischemic_heart_disease.gbd_cause)
    csmr_filter = csmr493.query("age == {a} and sex == 'Male' and year == {y}".format(a=age, y=year_start))
    cause_val = csmr_filter['rate'].values[0]

    cause_deleted = get_cause_deleted_mortality_rate(location_id=location_id,
                                                     year_start=year_start,
                                                     year_end=year_end,
                                                     list_of_csmrs=[csmr493],
                                                     gbd_round_id=gbd_round_id,
                                                     draw_number=draw_number)
    cause_deleted_filter = cause_deleted.query("age == {} and sex == 'Male' and year == {}".format(age, year_start))
    cause_deleted_val = cause_deleted_filter['cause_deleted_mortality_rate'].values[0]

    assert cause_deleted_val == all_cause_val - cause_val, "cause deleted mortality rate was incorrectly calculated"


def test_get_angina_proportions():
    props = get_angina_proportions()
    props.set_index('age', inplace=True)
    props = props.query('sex_id == 1')

    err_msg = ("get_angina_proportions needs to assign values for people younger "
               "than age group 9 to get the same value as people in age group 9")
    assert np.allclose(props.get_value(7.5, 'angina_prop'), props.get_value(22.5, 'angina_prop')), err_msg
    err_msg = "get_angina_proportions needs to return values that match input file"
    assert np.allclose(props.get_value(82.5, 'angina_prop'), 0.128526646), err_msg


def test_get_disability_weight():
    # me_id 2608 = mild diarrhea
    err_msg = ("Get_disability_weight should return the correct "
               "disability weight from the flat files prepared by central comp")
    assert np.allclose(get_disability_weight(dis_weight_modelable_entity_id=2608, draw_number=0), 0.0983228), err_msg


def test_get_asympt_ihd_proportions():
    angina_proportions = get_angina_proportions()
    heart_failure_proportions = get_post_mi_heart_failure_proportion_draws(180, 1990, 2000, draw_number=0)
    asympt_ihd_proportions = get_asympt_ihd_proportions(180, 1990, 2000, draw_number=0)

    ang_filter = angina_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")
    hf_filter = heart_failure_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")
    asy_filter = asympt_ihd_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")

    ang_value = ang_filter.set_index('age').get_value(32.5, 'angina_prop')
    hf_value = hf_filter.set_index('age').get_value(32.5, 'draw_19')
    asy_value = asy_filter.set_index('age').get_value(32.5, 'asympt_prop_19')

    assert 1 - hf_value - ang_value == asy_value, ("get_asympt_ihd_proportions needs to ensure that the sum of "
                                                   "heart failure, angina, and asympt ihd add up to 1")


@patch('ceam_inputs.gbd_ms_functions.get_populations')
@patch('ceam_inputs.gbd_ms_functions.gbd')
def test_assign_subregions_with_subregions(gbd_mock, get_populations_mock):
    gbd_mock.get_subregions.side_effect = lambda location_id: [10, 11, 12]
    test_populations = {
            10: build_table(20, ['age', 'year', 'sex', 'pop_scaled']),
            11: build_table(30, ['age', 'year', 'sex', 'pop_scaled']),
            12: build_table(50, ['age', 'year', 'sex', 'pop_scaled']),
    }
    get_populations_mock.side_effect = lambda location_id, year, sex, gbd_round_id: test_populations[location_id]

    locations = assign_subregions(pd.Index(range(100000)), 180, 2005, 3)

    counts = locations.value_counts()
    counts = np.array([counts[lid] for lid in [10, 11, 12]])
    counts = counts / counts.sum()
    assert np.allclose(counts, [.2, .3, .5], rtol=0.01)


@patch('ceam_inputs.gbd_ms_functions.get_populations')
@patch('ceam_inputs.gbd_ms_functions.gbd')
def test_assign_subregions_without_subregions(gbd_mock, get_populations_mock):
    gbd_mock.get_subregions.side_effect = lambda location_id: []
    test_populations = {
            190: build_table(100, ['age', 'year', 'sex', 'pop_scaled']),
    }
    get_populations_mock.side_effect = lambda location_id, year, sex: test_populations[location_id]

    locations = assign_subregions(pd.Index(range(1000)), 190, 2005, 3)

    assert np.all(locations == 190)


def test_get_etiology_specific_incidence():
    df = get_etiology_specific_incidence(180, 1990, 2000, 181, 302, 1181, gbd_round_id=3, draw_number=0)

    df = df.query("year_id == 1995 and sex_id ==1")

    val = df.set_index('age').get_value(82.5, 'draw_10')

    assert np.isclose(val, 0.06306237 * 2.5101927), ("get_etiology_specific_incidence needs to ensure the eti "
                                                     "pafs and envelope were multiplied together correctly")


def test_get_etiology_specific_prevalence():
    df = get_etiology_specific_prevalence(180, 1990, 2000, 181, 302, 1181, gbd_round_id=3, draw_number=0)

    df = df.query("year_id == 1995 and sex_id ==1")

    val = df.set_index('age').get_value(82.5, 'draw_10')

    assert np.allclose(val, (0.02491546 * 0.06306237)), ("get_etiology_specific_prevalence needs to ensure "
                                                         "the eti pafs and envelope were multiplied together correctly")


def test_get_severe_diarrhea_excess_mortality():
    # supply a dataframe with fake excess mortality data, fake severity split data,
    # and ensure that output is the rate divided by the severity split
    df = get_excess_mortality(1181)
    df['rate'] = 4
    severe_diarrhea_proportion = .5
    result = get_severe_diarrhea_excess_mortality(df, severe_diarrhea_proportion)

    assert np.allclose(result['rate'], 8), ("Get_severe_diarrhea_excess mortality should "
                                            "return the excess mortality rate for severe "
                                            "diarrhea cases only")


# get_rota_vaccine_coverage
def test_get_rota_vaccine_coverage():
    cov = get_rota_vaccine_coverage(180, 2000, 2015, 4)

    cov_filter1 = cov.query("year_id < 2014")
    cov1 = cov_filter1.draw_0.values
    cov_filter2 = cov.query("year_id > 2014 and age > 5")
    cov2 = cov_filter2.draw_0.values
    cov_filter3 = cov.query("year_id > 2014 and age <5")
    cov3 = cov_filter3.values

    assert not np.any(cov1), "this function should return an estimate of 0% coverage in Kenya for all ages before 2014"
    assert not np.any(cov2), "this function should return an estimate of 0% coverage for all people over the age of 5"
    assert np.all(cov3), ("this function should return an estimate of GT 0% coverage "
                          "for all people under the age of 5 after 2014")


def test_get_mediation_factors():
    assert np.isclose(get_mediation_factors(108, 493, 0), 0.53957565)
    assert np.isclose(get_mediation_factors(108, 499, 2), 0.48045271)
    assert np.isclose(get_mediation_factors(108, 500, 1), 0.54893082)
