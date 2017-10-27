import numpy as np
import pytest

from ceam_inputs import causes, risk_factors
from ceam_inputs.gbd_ms_functions import (get_sbp_mean_sd, get_relative_risks, get_pafs, get_exposures,
                                          get_angina_proportions, get_disability_weight,
                                          get_post_mi_heart_failure_proportion_draws,
                                          get_asympt_ihd_proportions,
                                          get_rota_vaccine_coverage, get_mediation_factors)

KENYA = 180

# TODO: Several tests here should be rewritten as unit tests on data transformations.  We
# also need to set up some actual data validation tests (probably to be run by our build system
# daily or weekly) to capture unexpected changes in the underlying data.


def test_get_relative_risks():
    df = get_relative_risks(location_id=KENYA,
                            risk_id=risk_factors.high_systolic_blood_pressure.gbd_id,
                            cause_id=causes.ischemic_heart_disease.gbd_id,
                            gbd_round_id=3,
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
                  risk_id=risk_factors.high_systolic_blood_pressure.gbd_id,
                  cause_id=causes.ischemic_heart_disease.gbd_id,
                  gbd_round_id=3,
                  paf_type='morbidity')
    draw_number = 19
    df = df[df.year_id == 1990]
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
                       risk_id=risk_factors.smoking_prevalence_approach.gbd_id,
                       gbd_round_id=3)
    df = df[df.year_id == 1990]
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
    df = get_sbp_mean_sd(location_id=KENYA, gbd_round_id=3)
    df = df[df.year_id == 1990]
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
    df = get_sbp_mean_sd(location_id=KENYA, gbd_round_id=3)
    df = df[df.year_id == 2000]
    df = df[['year_id', 'sex_id', 'age', 'log_mean_0']]
    df = df.groupby(['year_id', 'sex_id', 'age']).first()

    err_msg = ('should match data loaded by @aflaxman on 8/4/2016. test changed by '
               '@emumford on 9/23 to account for change in gbd_ms_functions')
    assert np.allclose(df.loc[(2000, 1, 27.5), 'log_mean_0'], np.log(118.948299)), err_msg


def test_get_angina_proportions():
    props = get_angina_proportions(gbd_round_id=3)
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
    assert np.allclose(get_disability_weight(causes.diarrhea.severity_splits.mild,
                                             draw_number=0,
                                             gbd_round_id=3), 0.0983228), err_msg


def test_get_asympt_ihd_proportions(base_config):
    angina_proportions = get_angina_proportions(gbd_round_id=3)
    pub_ids = base_config.input_data.gbd_publication_ids
    heart_failure_proportions = get_post_mi_heart_failure_proportion_draws(KENYA,
                                                                           gbd_round_id=3)
    asympt_ihd_proportions = get_asympt_ihd_proportions(KENYA,
                                                        gbd_round_id=3)

    ang_filter = angina_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")
    hf_filter = heart_failure_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")
    asy_filter = asympt_ihd_proportions.query("age == 32.5 and sex_id == 1 and year_id==1995")

    ang_value = ang_filter.set_index('age').get_value(32.5, 'angina_prop')
    hf_value = hf_filter.set_index('age').get_value(32.5, 'draw_19')
    asy_value = asy_filter.set_index('age').get_value(32.5, 'asympt_prop_19')

    assert 1 - hf_value - ang_value == asy_value, ("get_asympt_ihd_proportions needs to ensure that the sum of "
                                                   "heart failure, angina, and asympt ihd add up to 1")


# get_rota_vaccine_coverage
def test_get_rota_vaccine_coverage():
    cov = get_rota_vaccine_coverage(location_id=180, gbd_round_id=4)

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
    assert np.isclose(get_mediation_factors(risk_id=108, cause_id=493, gbd_round_id=3, draw_number=0), 0.53957565)
    assert np.isclose(get_mediation_factors(risk_id=108, cause_id=499, gbd_round_id=3, draw_number=2), 0.48045271)
    assert np.isclose(get_mediation_factors(risk_id=108, cause_id=500, gbd_round_id=3, draw_number=1), 0.54893082)
